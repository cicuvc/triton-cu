from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, nvidia
from triton import knobs
from triton.runtime.errors import PTXASError

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import glob
import subprocess
from pathlib import Path


def _find_cuda_include():
    cuda_home = os.environ.get("CUDA_HOME", "")
    if cuda_home:
        inc = os.path.join(cuda_home, "targets", "x86_64-linux", "include")
        if os.path.isdir(inc):
            return inc
    for cuda_base in ["/usr/local/cuda", "/opt/cuda"]:
        matches = sorted(glob.glob(cuda_base + "*"))
        for m in matches:
            inc = os.path.join(m, "targets", "x86_64-linux", "include")
            if os.path.isdir(inc):
                return inc
    raise FileNotFoundError(
        "Cannot find CUDA include dir. Set CUDA_HOME env var "
        "or install CUDA under /usr/local/cuda.")


def _find_clang_resource_dir():
    llvm_path = os.environ.get("LLVM_SYSPATH", "")
    if llvm_path:
        clang_dir = os.path.join(llvm_path, "lib", "clang")
        if os.path.isdir(clang_dir):
            versions = sorted(os.listdir(clang_dir))
            for v in versions:
                rd = os.path.join(clang_dir, v)
                if os.path.isdir(rd):
                    return rd

    # Fallback: check llvm-project/install relative to repo root
    file_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = file_dir
    for _ in range(5):
        repo_root = os.path.dirname(repo_root)
    for prefix in [
        os.path.join(repo_root, "..", "llvm-project", "install"),
        os.path.join(repo_root, "llvm-project", "install"),
        os.path.join(file_dir, "..", "..", "..", "..", "llvm-project", "install"),
    ]:
        clang_dir = os.path.join(prefix, "lib", "clang")
        if os.path.isdir(clang_dir):
            versions = sorted(os.listdir(clang_dir))
            for v in versions:
                rd = os.path.join(clang_dir, v)
                if os.path.isdir(rd):
                    return rd

    raise FileNotFoundError(
        "Cannot find clang resource dir. Set LLVM_SYSPATH env var.")


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:  # [m, n, k]
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        # For small M/N the input we can still use tensorcores with padding.
        if lhs_bitwidth == 8:
            return (1, 1, 32)
        elif lhs_bitwidth == 64:
            return (1, 1, 4)
        elif lhs_bitwidth == 32:
            return (1, 1, 8)
        else:
            return (1, 1, 16)

    return check_dot_compatibility


def get_ptxas(arch: int) -> knobs.NvidiaTool:
    return knobs.nvidia.ptxas_blackwell if arch >= 100 else knobs.nvidia.ptxas


@functools.lru_cache()
def get_ptxas_version(arch: int = 80):
    mock_ver = knobs.nvidia.mock_ptx_version
    if mock_ver is not None:
        return mock_ver  # This is not really a version of ptxas, but it is good enough for testing
    version = subprocess.check_output([get_ptxas(arch).path, "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor

    if major >= 13:
        base_ptx = 90
        return base_ptx + (major - 13) * 10 + minor

    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)


def get_ptx_version_from_options(options, arch: int):
    ptx_version = options.ptx_version
    if ptx_version is None:
        cuda_version = get_ptxas(arch).version
        ptx_version = ptx_get_version(cuda_version)
    return ptx_version


@functools.lru_cache()
def get_features(options, arch: int):
    ptx_version = get_ptx_version_from_options(options, arch)

    # PTX 8.6 is the max version supported by llvm 979132a0.
    #
    # To check if a newer PTX version is supported, increase this value
    # and run a test.  If it's not supported, LLVM will print a warning
    # like "+ptx8.4 is not a recognized feature for this target".
    llvm_ptx_version = min(90, ptx_version)
    features = f'+ptx{llvm_ptx_version}'
    return features


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sm_arch_from_capability(capability: int):
    # TODO: Handle non-"a" sms
    suffix = "a" if capability >= 90 else ""
    return f"sm_{capability}{suffix}"


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 32
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    ptx_version: int = None
    ptx_options: Optional[str] = knobs.nvidia.ptxas_options
    ir_override: Optional[str] = None  # filename of a user-defined IR (*.{ttir|ttgir|llir|ptx})
    enable_fp_fusion: bool = True
    enable_reflect_ftz: bool = True  # ftz in libdevice
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee", 'bf16x3', 'bf16x6')
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'cuda'
    sanitize_overflow: bool = True
    arch: str = None
    instrumentation_mode: str = ""

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = knobs.nvidia.libdevice_path or str(default_libdir / 'libdevice.10.bc')
        if "gsan" in self.instrumentation_mode:
            gsan_lib = default_libdir / "gsan.ll"
            if not gsan_lib.exists():
                raise FileNotFoundError(f"GSan runtime is missing at {gsan_lib}. "
                                        "Rebuild Triton to generate it.")
            extern_libs['gsan'] = str(gsan_lib)

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @property
    def enable_iisan(self):
        return "iisan" in self.instrumentation_mode


def _serialize_return_types(return_type_map):
    """Serialize return type map (symbol -> list of TensorParameter) to JSON dict."""
    scalar_names = {
        llvm.ScalarType.Fp32: "f32", llvm.ScalarType.Fp16: "f16",
        llvm.ScalarType.Bf16: "bf16", llvm.ScalarType.Int32: "i32",
        llvm.ScalarType.Int64: "i64",
    }
    result = {}
    for symbol, tp_list in return_type_map.items():
        result[symbol] = [{
            "scalar": scalar_names.get(tp.type, "f32"),
            "shape": list(tp.shape),
            "layout_shape": list(tp.layout_shape),
            "reg_basis": list(tp.reg_basis),
            "lane_basis": list(tp.lane_basis),
            "warp_basis": list(tp.warp_basis),
            "n_warps": tp.n_warps,
        } for tp in tp_list]
    return result


# D-01/D-02: Inference hook object exposed via codegen_fns.
# Created at get_codegen_implementation time with compiler construction
# params.  make_ir creates suspended CUDACompilers at semantic time;
# _pre_compile_extern_calls resumes them at the llir stage to emit
# bitcode without a second clang parse.
class InferExternCallResult:

    def __init__(self, sm, resource_dir, include_paths):
        import triton._C.libtriton.llvm as _llvm
        self._sm = sm
        self._resource_dir = resource_dir
        self._include_paths = include_paths
        self._compilers = {}  # libpath -> SuspendedCudaCompiler
        self._llvm_ctx = None  # Shared LLVMContext (set on first create_and_suspend)
        # D-07: Snapshot the parse counter at hook-creation time so the
        # per-compile delta spans both semantic and llir stages.
        self._parse_count_before = _llvm.get_extern_cuda_parse_count()

    def create_and_suspend(self, source, llvm_context, libpath):
        """Create CUDACompiler, parse+suspend it (parks in HandleTranslationUnit).

        Called at semantic time by Gluon make_ir for each distinct .cu file.
        The compiler's coroutine is parked waiting for tasks — ASTContext alive.

        `llvm_context` must be an LLVM context (from llvm.context()), shared
        across the entire compile so bitcode links correctly at the llir stage.
        """
        import triton._C.libtriton.llvm as llvm
        if self._llvm_ctx is None:
            self._llvm_ctx = llvm_context
        compiler = llvm.SuspendedCudaCompiler(
            source, 3, self._sm, self._resource_dir, self._include_paths)
        compiler.parse(llvm_context, "cudamod")
        self._compilers[libpath] = compiler

    def infer_result(self, libpath, func, arg_params, use_fast_math):
        """Infer the CUDA return type (scalar, shape) at semantic time.

        Uses PlaceholderLayout-based probing to determine dtype+shape
        without requiring concrete layouts. Returns a list of
        (scalar_name, shape) tuples, one per result.

        Args:
            libpath: Path to the .cu source file (key into _compilers dict).
            func: CUDA device function name.
            arg_params: List of dicts with "dtype" and "shape" keys.
            use_fast_math: Whether to use fast-math optimizations.

        Returns:
            List[Tuple[str, List[int]]]: Per-result (scalar_name, shape) tuples.
        """
        import triton._C.libtriton.llvm as llvm

        compiler = self._compilers.get(libpath)
        if compiler is None:
            raise RuntimeError(
                f"infer_result: no suspended compiler for {libpath} — "
                f"make_ir must create a SuspendedCudaCompiler first")

        # dtype→ScalarType mapping (mirrors _scalar_type_for in
        # _pre_compile_extern_calls, lines 648-664)
        _dtype_to_scalar = {
            "f32": llvm.ScalarType.Fp32, "fp32": llvm.ScalarType.Fp32,
            "f16": llvm.ScalarType.Fp16, "fp16": llvm.ScalarType.Fp16,
            "bf16": llvm.ScalarType.Bf16,
            "i32": llvm.ScalarType.Int32, "s32": llvm.ScalarType.Int32,
            "i64": llvm.ScalarType.Int64, "s64": llvm.ScalarType.Int64,
        }

        def _scalar_type_for(dtype_str):
            st = _dtype_to_scalar.get(dtype_str)
            if st is None:
                if dtype_str in ("f64", "fp64", "float64"):
                    raise NotImplementedError(
                        "gl.call() does not support float64; full Fp64 "
                        "support is out of scope (see FP64-01)")
                raise ValueError(f"Unsupported dtype: {dtype_str}")
            return st

        # Build CudaFuncRequest — same representation as
        # _pre_compile_extern_calls (lines 684-702)
        req = llvm.CudaFuncRequest()
        req.symbol = func
        req.use_fast_math = use_fast_math

        param_types = []
        for ap in arg_params:
            if ap.get("memory_space") == "shared":
                # D-12: Degenerate all-zero SharedTensorParameter for
                # template deduction. Only T + Shape matter; real bases
                # flow at the llir stage from Phase-5 extracted specs.
                stp = llvm.SharedTensorParameter()
                stp.type = _scalar_type_for(ap["dtype"])
                stp.shape = ap["shape"]
                stp.offset_basis = []       # degenerate: all-zero bases
                stp.block_basis = []        # degenerate: all-zero bases
                stp.alignment = 16          # default alignment
                stp.layout_rank = len(ap["shape"])
                param_types.append(stp)
            elif ap.get("scalar") is not None:
                # Scalar arg — pass as ScalarType for plain T params (e.g. T scale)
                param_types.append(_scalar_type_for(ap["scalar"]))
            else:
                tp = llvm.TensorParameter()
                tp.type = _scalar_type_for(ap["dtype"])
                tp.shape = ap["shape"]
                tp.layout_shape = ap["shape"]
                # Compute minimally valid concrete bases for dtype+shape
                # inference. The exact layout doesn't matter — we just need a
                # valid Tensor<T,Shape,Layout<...>> type for template deduction.
                # N_WARPS=1, all-zero bases produce a degenerate but valid layout.
                rank = len(ap["shape"])
                size = 1
                for d in ap["shape"]:
                    size *= int(d)
                n_warps = 1
                # ffs(x) = bit_length of lowest set bit; equivalently (x & -x).bit_length()
                _ffs_size_per_warp = (size // n_warps)
                _lsb_bit = (_ffs_size_per_warp & -_ffs_size_per_warp).bit_length() if _ffs_size_per_warp > 0 else 0
                n_lane_axes = 5
                n_reg_axes = max(0, _lsb_bit - n_lane_axes - 1)
                n_warp_axes = max(0, (n_warps & -n_warps).bit_length() - 1)
                tp.reg_basis = [0] * (n_reg_axes * rank)
                tp.lane_basis = [0] * (n_lane_axes * rank)
                tp.warp_basis = [0] * (n_warp_axes * rank)
                tp.n_warps = n_warps
                param_types.append(tp)
        req.param_types = param_types

        # Call inference-only C++ method (D-01)
        ok, bitcode, error, results = compiler.infer([req])
        if not ok:
            raise RuntimeError(
                f"infer_result: CUDA type inference failed for "
                f"'{func}': {error}")

        _scalar_names = {
            llvm.ScalarType.Fp32: "f32", llvm.ScalarType.Fp16: "f16",
            llvm.ScalarType.Bf16: "bf16", llvm.ScalarType.Int32: "i32",
            llvm.ScalarType.Int64: "i64",
        }
        inferred = []
        for result_tuple in results:
            symbol, _, ret_types, _ = result_tuple
            for tp in ret_types:
                scalar_name = _scalar_names.get(tp.type, "f32")
                inferred.append((scalar_name, list(tp.shape)))

        if len(inferred) == 0:
            return []  # void-returning functions produce no results
        return inferred

    def compile_bitcode(self, libpath, requests):
        """Resume the suspended compiler and emit device bitcode.

        Called at the llir stage by _pre_compile_extern_calls.
        Returns (bitcode_bytes, mangled_names_dict, extractor_names_dict, ret_types_list).
        """
        compiler = self._compilers.get(libpath)
        if compiler is None:
            raise RuntimeError(
                f"No suspended compiler for {libpath} — "
                f"must be created by make_ir at semantic time")
        ok, bitcode, error, results = compiler.compile_bitcode(requests)
        if not ok:
            raise RuntimeError(
                f"In-process CUDA compilation failed for {libpath}: {error}")
        mangled_names = {}
        extractor_names = {}
        ret_types_list = []
        for symbol, mangled, ret_types, extr_names in results:
            mangled_names[symbol] = mangled
            if ret_types:
                ret_types_list.append((symbol, list(ret_types)))
            if extr_names:
                extractor_names[symbol] = list(extr_names)
        return bitcode, mangled_names, extractor_names, ret_types_list


class CUDABackend(BaseBackend):
    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cuda'

    def _parse_arch(self, arch):
        pattern = r"^sm(\d+)$"
        match = re.fullmatch(pattern, arch)
        if not match:
            raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
        return int(match.group(1))

    def get_target_name(self, options) -> str:
        capability = self._parse_arch(options.arch)
        return f"cuda:{capability}"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        # Enable debug mode for ConSan, so device-side assertions are not optimized out
        if any(mode in opts.get("instrumentation_mode", "") for mode in ["consan", "iisan"]):
            opts["debug"] = True
            opts["sanitize_overflow"] = False

        args = {'arch': knobs.runtime.override_arch or f"sm{self.target.arch}"}
        args.update({k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        capability = int(self._parse_arch(args["arch"]))

        if args.get("num_ctas", 1) > 1 and capability < 90:
            raise ValueError((f"num_ctas > 1 requires NVIDIA SM90+ (Hopper). "
                              f"Current target is sm_{capability}. This configuration will fail. "
                              f"Please set num_ctas=1 or target an SM90+ GPU."))

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CUDAOptions.supported_fp8_dtypes)
            if capability >= 89:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dot_operand_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dot_operand_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0

        return CUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        import triton.language.extra.cuda as cuda
        import os as _os
        capability = int(self._parse_arch(options.arch))
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if capability >= 80 else cuda.convert_custom_float8_sm70, "min_dot_size":
            min_dot_size(self.target)
        }

        # D-01: Build the inference hook object with compiler construction
        # params.  These mirror _pre_compile_extern_calls setup (minus the
        # per-libpath source which is not available here).
        _resource_dir = _find_clang_resource_dir()
        _cuda_inc = os.path.join(os.path.dirname(__file__), "include")
        _sm = f"sm_{capability // 10}{capability % 10}"
        _include_paths = [_cuda_inc, _find_cuda_include()]

        _infer_hook = InferExternCallResult(_sm, _resource_dir, _include_paths)
        self._infer_hook = _infer_hook  # stored for _pre_compile_extern_calls
        codegen_fns["infer_extern_call_result"] = _infer_hook
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        nvidia.load_dialects(ctx)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        if capability // 10 < 9:
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        # Set maxnreg on all kernels, if it was provided.
        if opt.maxnreg is not None:
            mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

        pm = ir.pass_manager(mod.context)
        dump_enabled = pm.enable_debug()
        emuTF32 = (capability // 10 >= 8)
        passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        nvidia.passes.ttnvgpuir.add_plan_cta(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
        passes.ttir.add_loop_aware_cse(pm)
        if capability // 10 in [8, 9]:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        elif capability // 10 >= 10:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.ttgpuir.add_optimize_accumulator_init(pm)
            passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
            nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_optimize_partition_warps(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            # hoist again and allow hoisting out of if statements
            passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
            nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
        else:
            passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)
        if capability // 10 == 8:
            passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_coalesce_async_copy(pm)
        nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
        if capability // 10 >= 9:
            nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
        nvidia.passes.ttnvgpuir.add_lower_mma(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        if "fpsan" in opt.instrumentation_mode:
            passes.ttgpuir.add_fp_sanitizer(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)

        pm.run(mod, 'make_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def gluon_to_ttgir(self, src, metadata, options, capability):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_infer_coalesced_encodings(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        if "fpsan" in options.instrumentation_mode:
            passes.ttgpuir.add_fp_sanitizer(pm)
        if any(mode in options.instrumentation_mode for mode in ["consan", "fpsan"]):
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)

        pm.run(mod, 'gluon_to_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def make_llir(self, src, metadata, options, capability):
        ptx_version = get_ptx_version_from_options(options, self.target.arch)

        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        if "gsan" in options.instrumentation_mode:
            # GSan introduces layout conversions, so must come before shared memory allocation
            passes.ttgpuir.add_global_sanitizer(pm)

        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm, "consan" in options.instrumentation_mode)
        passes.convert.add_scf_to_cf(pm)
        passes.gluon.add_inliner(pm)
        if "consan" in options.instrumentation_mode:
            passes.ttgpuir.add_prepare_consan_captures(pm, "nvidia")
        nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
        nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
        nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)
        nvidia.passes.ttnvgpuir.add_tmem_barrier_insertion(pm)
        nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version, "consan" in options.instrumentation_mode)
        nvidia.passes.ttnvgpuir.add_initialize_ws_cluster_barriers(pm, capability, ptx_version)
        passes.ttgpuir.add_canonicalize_llvm_ir(pm)
        passes.common.add_cse(pm)
        nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
        nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.convert.add_nvvm_to_llvm(pm)

        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

        # Pre-compile extern calls before MLIR lowering.
        # This instantiates template device functions and produces
        # mangled name mappings that ExternCallOpToLLVM reads.
        import json as _json
        llvm.init_targets()
        # D-05: Reuse the semantic-stage LLVMContext if the hook created one
        # (from make_ir / create_and_suspend).  Otherwise create a fresh one.
        _hook = getattr(self, '_infer_hook', None)
        _shared_ctx = _hook._llvm_ctx if _hook is not None else None
        context = _shared_ctx if _shared_ctx is not None else llvm.context()
        has_extern_calls = self._pre_compile_extern_calls(
            mod, metadata, capability, context)
        if has_extern_calls:
            mod.set_str_attr("ttg.extern_call_mangled_names",
                             _json.dumps(metadata["extern_call_mangled"]))
            if metadata.get("extern_call_extractor_names"):
                mod.set_str_attr("ttg.extern_call_extractor_names",
                                 _json.dumps(metadata["extern_call_extractor_names"]))
            # D-16: Per-symbol per-arg memory-space list for the C++ lowering.
            if metadata.get("extern_call_arg_spaces"):
                mod.set_str_attr("ttg.extern_call_arg_spaces",
                                 _json.dumps(metadata["extern_call_arg_spaces"]))

        pm.run(mod, 'make_llir')

        if knobs.compilation.dump_ir_extract_di_local_variables:
            if not knobs.compilation.disable_line_info:
                pm = ir.pass_manager(mod.context)
                pm.enable_debug()
                passes.llvmir.add_di_scope(pm)
                pm.run(mod, 'make_llir.disable_line_info')

            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_local_variable(pm)
            pm.run(mod, 'make_llir.dump_ir_extract_di_local_variables')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        if knobs.compilation.enable_asan:
            raise RuntimeError(
                "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
        llvm_mod = llvm.to_module(mod, context)
        if not llvm.verify_module(llvm_mod):
            raise RuntimeError("LLVM module verification failed after MLIR translation")
        proc = sm_arch_from_capability(capability)
        features = get_features(options, self.target.arch)
        triple = 'nvptx64-nvidia-cuda'
        nvidia.set_short_ptr()
        llvm.attach_datalayout(llvm_mod, triple, proc, features)
        if options.enable_reflect_ftz:
            nvidia.set_nvvm_reflect_ftz(llvm_mod)

        if options.extern_libs and nvidia.has_extern_deps(llvm_mod):
            paths = [path for (name, path) in options.extern_libs
                     if not name.startswith("__extern_call_src_")]
            if paths:
                llvm.link_extern_libs(llvm_mod, paths)

        # Link pre-compiled extern_call CUDA bitcodes (same LLVMContext).
        if has_extern_calls:
            bitcodes = metadata.get("extern_call_bitcodes", [])
            for bc in bitcodes:
                llvm.link_cuda_bitcode(llvm_mod, bytes(bc), context)

            if not llvm.verify_module(llvm_mod):
                raise RuntimeError("LLVM module verification failed after extern linking")

        # D-07: Per-compile parse-count assertion — guards against
        # double-parsing.  The delta (parses during THIS compile, from
        # hook creation through now) must equal the number of distinct
        # .cu files.
        if has_extern_calls:
            parse_count_delta = metadata.get("extern_parse_delta", 0)
            distinct_cu = metadata.get("extern_distinct_cu", 0)
            assert parse_count_delta == distinct_cu, (
                f"extern CUDA parse count mismatch: {parse_count_delta} parse(s) "
                f"for {distinct_cu} distinct .cu file(s) "
                f"(per-compile delta; must survive multiple compiles in one process)")

        # Run optimization
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3,
                             disable_slp_vectorizer=capability == 80)
        

        # Get some metadata
        # warp-specialization mutates num_warps
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size") or 0
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment") or 1
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    def _pre_compile_extern_calls(self, mod, metadata, capability, llvm_context):
        """Pre-compile extern_call ops before MLIR lowering.
        Scans MLIR module for ttg.extern_call ops, builds template function
        instantiation specs, compiles CUDA sources in-process, and stores
        mangled name mappings for the lowering pass to use."""
        import json as _json
        import os

        json_str = llvm.extract_extern_call_specs(mod)
        if json_str is None or json_str == "[]":
            return False

        specs_list = _json.loads(json_str)

        # D-05: If the semantic stage already created a shared LLVMContext
        # (via make_ir / InferExternCallResult.create_and_suspend), reuse
        # it so bitcode from suspended compilers links correctly. Otherwise
        # fall back to the llir-stage context.
        _hook = getattr(self, '_infer_hook', None)
        if _hook is not None and _hook._llvm_ctx is not None:
            llvm_context = _hook._llvm_ctx

        resource_dir = _find_clang_resource_dir()
        cuda_inc = os.path.join(os.path.dirname(__file__), "include")
        sm = f"sm_{capability // 10}{capability % 10}"
        _include_paths = [cuda_inc, _find_cuda_include()]

        dtype_to_scalar = {
            "f32": llvm.ScalarType.Fp32, "fp32": llvm.ScalarType.Fp32,
            "f16": llvm.ScalarType.Fp16, "fp16": llvm.ScalarType.Fp16,
            "bf16": llvm.ScalarType.Bf16,
            "i32": llvm.ScalarType.Int32, "s32": llvm.ScalarType.Int32,
            "i64": llvm.ScalarType.Int64, "s64": llvm.ScalarType.Int64,
        }


        def _scalar_type_for(dtype_str):
            st = dtype_to_scalar.get(dtype_str)
            if st is None:
                if dtype_str in ("f64", "fp64", "float64"):
                    raise NotImplementedError(
                        "gl.call() does not support float64; full Fp64 support is out of scope (see FP64-01)")
                raise ValueError(f"Unsupported dtype: {dtype_str}")
            return st

        # Group by libpath
        by_libpath = {}
        for spec in specs_list:
            libpath = spec["libpath"]
            by_libpath.setdefault(libpath, []).append(spec)

        compiled_bitcodes = []
        mangled_names = {}  # dict[original_symbol] = mangled_name
        return_type_map = {}  # dict[original_symbol] = list[TensorParameter]
        extractor_names = {}  # dict[original_symbol] = list[str]

        for libpath, specs in by_libpath.items():
            # D-05/D-06: Prefer suspended compiler (created by make_ir at
            # semantic time).  If a suspended compiler exists for this .cu,
            # resume it via the hook's compile_bitcode — no second parse.
            _hook = getattr(self, '_infer_hook', None)
            _suspended = _hook._compilers.get(libpath) if _hook is not None else None
            if _suspended is not None:
                requests = []
                for spec_entry in specs:
                    symbol = spec_entry["symbol"]
                    req = llvm.CudaFuncRequest()
                    req.symbol = symbol
                    req.use_fast_math = spec_entry.get("use_fast_math", False)

                    param_types = []
                    for inp in spec_entry["inputs"]:
                        if inp.get("memory_space") == "shared":
                            stp = llvm.SharedTensorParameter()
                            stp.type = _scalar_type_for(inp["dtype"])
                            stp.shape = inp["shape"]
                            stp.offset_basis = inp.get("offset_bases", [])
                            stp.block_basis = inp.get("block_bases", [])
                            stp.alignment = inp.get("alignment", 16)
                            stp.layout_rank = len(inp["shape"])
                            param_types.append(stp)
                        elif inp.get("scalar") is not None:
                            param_types.append(_scalar_type_for(inp["scalar"]))
                        else:
                            tp = llvm.TensorParameter()
                            tp.type = _scalar_type_for(inp["dtype"])
                            tp.shape = inp["shape"]
                            tp.layout_shape = inp["shape"]
                            tp.reg_basis = inp.get("reg_bases", [])
                            tp.lane_basis = inp.get("lane_bases", [])
                            tp.warp_basis = inp.get("warp_bases", [])
                            tp.n_warps = inp.get("num_warps", 1)
                            param_types.append(tp)
                    req.param_types = param_types
                    requests.append(req)

                bitcode, mangled_names_batch, extractor_names_batch, ret_types_list = \
                    _hook.compile_bitcode(libpath, requests)
                compiled_bitcodes.append(list(bitcode))
                mangled_names.update(mangled_names_batch)
                extractor_names.update(extractor_names_batch)
                for symbol, ret_types in ret_types_list:
                    return_type_map[symbol] = ret_types
                continue  # skip the old compile_cuda_to_module path

            # Fallback: existing compile_cuda_to_module path (for .cu files
            # not covered by the suspended-compiler pre-scan).
            with open(libpath) as f:
                source = f.read()

            requests = []
            for spec_entry in specs:
                symbol = spec_entry["symbol"]
                req = llvm.CudaFuncRequest()
                req.symbol = symbol
                req.use_fast_math = spec_entry.get("use_fast_math", False)

                param_types = []
                for inp in spec_entry["inputs"]:
                    if inp.get("memory_space") == "shared":
                        stp = llvm.SharedTensorParameter()
                        stp.type = _scalar_type_for(inp["dtype"])
                        stp.shape = inp["shape"]
                        stp.offset_basis = inp.get("offset_bases", [])
                        stp.block_basis = inp.get("block_bases", [])
                        stp.alignment = inp.get("alignment", 16)
                        stp.layout_rank = len(inp["shape"])
                        param_types.append(stp)
                    else:
                        tp = llvm.TensorParameter()
                        tp.type = _scalar_type_for(inp["dtype"])
                        tp.shape = inp["shape"]
                        tp.layout_shape = inp["shape"]
                        tp.reg_basis = inp.get("reg_bases", [])
                        tp.lane_basis = inp.get("lane_bases", [])
                        tp.warp_basis = inp.get("warp_bases", [])
                        tp.n_warps = inp.get("num_warps", 1)
                        param_types.append(tp)
                req.param_types = param_types
                requests.append(req)

            ok, bitcode, error, results = llvm.compile_cuda_to_module(
                llvm_context, source, sm, resource_dir, _include_paths,
                requests)

            if not ok:
                raise RuntimeError(
                    f"In-process CUDA compilation failed for {libpath}: {error}")

            compiled_bitcodes.append(list(bitcode))  # bytes → list of ints for JSON

            for symbol, mangled, ret_types, extr_names in results:
                mangled_names[symbol] = mangled
                if ret_types:
                    return_type_map[symbol] = list(ret_types)
                if extr_names:
                    extractor_names[symbol] = list(extr_names)

        # Store inferred return types as module attribute for the
        # ExternCall lowering to use when building LLVM result types.
        if return_type_map:
            error = llvm.patch_extern_call_result_types(
                mod, _json.dumps(_serialize_return_types(return_type_map)))
            if error:
                raise RuntimeError(error)

        # D-16: Build per-symbol per-arg memory-space lists for the C++ lowering.
        arg_spaces_map = {}
        for spec in specs_list:
            symbol = spec["symbol"]
            spaces = []
            for inp in spec["inputs"]:
                spaces.append("shared" if inp.get("memory_space") == "shared" else "register")
            arg_spaces_map[symbol] = spaces

        metadata["extern_call_bitcodes"] = compiled_bitcodes
        metadata["extern_call_mangled"] = mangled_names
        metadata["extern_call_extractor_names"] = extractor_names
        metadata["extern_call_arg_spaces"] = arg_spaces_map

        # D-07: Per-compile delta — store the number of clang parses that
        # occurred during THIS compile only (from hook-creation time through
        # now, spanning both semantic and llir stages).
        _hook = getattr(self, '_infer_hook', None)
        _count_before = _hook._parse_count_before if _hook is not None else 0
        _parse_delta = llvm.get_extern_cuda_parse_count() - _count_before
        _distinct_cu = len(by_libpath)
        metadata["extern_parse_delta"] = _parse_delta
        metadata["extern_distinct_cu"] = _distinct_cu
        return True


    def make_ptx(self, src, metadata, opt, capability):
        ptx_version = get_ptx_version_from_options(opt, self.target.arch)

        triple = 'nvptx64-nvidia-cuda'
        proc = sm_arch_from_capability(capability)
        features = get_features(opt, self.target.arch)
        flags = ["nvptx-mad-wide-opt"]
        canonicalize_gep = "fpsan" in opt.instrumentation_mode
        ret = llvm.translate_to_asm(src, triple, proc, features, flags, opt.enable_fp_fusion, False, canonicalize_gep)
        # Find kernel names (there should only be one)
        names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]
        # post-process
        ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret, flags=re.MULTILINE)
        if not knobs.compilation.dump_ir_extract_di_local_variables:
            # Remove the debug flag that prevents ptxas from optimizing the code
            # Note: if this flag is removed, the source var name and type info will be lost when ptx was compiled into cubin
            #           and we may not be able to see them in cuda-gdb
            ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        if knobs.nvidia.dump_nvptx:
            print("// -----// NVPTX Dump //----- //")
            print(ret)
        return ret

    def make_cubin(self, src, metadata, opt, capability):
        ptxas = get_ptxas(self.target.arch).path
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            debug_info = []
            if knobs.compilation.disable_line_info:
                # This option is ignored if used without -lineinfo
                debug_info += ["-lineinfo", "-suppress-debug-info"]
            elif knobs.nvidia.disable_ptxas_opt:
                # Synthesize complete debug info
                debug_info += ["-g"]
            else:
                # Only emit line info
                debug_info += ["-lineinfo"]

            fmad = [] if opt.enable_fp_fusion else ["--fmad=false"]
            arch = sm_arch_from_capability(capability)

            # Disable ptxas optimizations if requested
            disable_opt = ['--opt-level', '0'] if knobs.nvidia.disable_ptxas_opt else []

            # Accept more ptxas options if provided
            ptx_extra_options = opt.ptx_options.split(" ") if opt.ptx_options else []

            # -Ofc mid miscompiles some large ConSan kernels into invalid global
            # accesses; -O1 keeps compile time reasonable without that ptxas bug.
            if (not knobs.nvidia.disable_ptxas_opt
                    and any(mode in opt.instrumentation_mode for mode in ["consan", "fpsan"])):
                ptx_extra_options += ["--opt-level", "1"]

            # Add --regAllocOptLevel=2 to work around ptxas 13.x bug
            reg_alloc = ['--regAllocOptLevel=2']

            ptxas_cmd = [
                ptxas, *debug_info, *fmad, '-v', *disable_opt, *reg_alloc, *ptx_extra_options, f'--gpu-name={arch}',
                fsrc.name, '-o', fbin
            ]
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if knobs.nvidia.dump_ptxas_log:
                    with open(flog.name) as log_file:
                        print(log_file.read())

                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                if e.returncode == 255:
                    error = 'Internal Triton PTX codegen error'
                elif e.returncode == 128 + signal.SIGSEGV:
                    error = '`ptxas` raised SIGSEGV'
                else:
                    error = f'`ptxas` failed with error code {e.returncode}'

                error = (f"{error}\n"
                         f"`ptxas` stderr:\n{log}\n"
                         f'Repro command: {" ".join(ptxas_cmd)}\n')

                print(f"""

================================================================
{error}

{src}
================================================================
please share the reproducer above with Triton project.
""")
                raise PTXASError(error)

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

    def add_stages(self, stages, options, language):
        capability = self._parse_arch(options.arch)
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options, capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.target.arch)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, capability)

    @functools.lru_cache()
    def hash(self):
        version = get_ptxas_version(self.target.arch)
        return f'{version}-{self.target.arch}'
