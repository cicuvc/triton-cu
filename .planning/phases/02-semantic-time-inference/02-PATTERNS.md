# Phase 2: Semantic-Time Inference - Pattern Map

**Mapped:** 2026-07-11
**Files analyzed:** 6 (to be modified)
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `python/triton/experimental/gluon/language/_semantic.py` (call_extern) | controller/semantic handler | request-response (builds MLIR ops) | `_semantic.py:250-279` (existing call_extern) | exact — same function |
| `third_party/nvidia/backend/compiler.py` (infer_result) | service/hook | request-response (Python → C++ → Python) | `compiler.py:228-252` (compile_bitcode in same class) | role-match — same InferExternCallResult class |
| `python/src/clang_compiler.cc` (inferReturnType) | service/compiler backend | request-response (CudaFuncRequest → CudaFuncResult) | `clang_compiler.cc:809-936` (compileBitcode) | exact — same CUDACompiler class, same coroutine pattern |
| `python/src/clang_compiler.h` (inferReturnType decl) | header declaration | N/A | `clang_compiler.h:348-349` (compileBitcode decl) | exact — adjacent declaration |
| `python/src/llvm.cc` (SuspendedCudaCompiler.infer) | Python binding | request-response (pybind11 wrapper) | `llvm.cc:1041-1066` (compile_bitcode binding) | exact — same py::class_, adjacent method |
| `python/test/gluon/tt_plugin.cu` (PlaceholderLayout) | library/CUDA template framework | N/A (type definitions) | `tt_plugin.cu:65-78` (Layout template) | role-match — same file, same template pattern |

## Pattern Assignments

### 1. `python/triton/experimental/gluon/language/_semantic.py` — `call_extern` (controller, request-response)

**Analog:** `_semantic.py:250-279` (existing `call_extern` — same function, Phase 2 adds hook consumption)

**Imports pattern** (lines 1-8):
```python
from typing import Sequence, List, TypeVar, Tuple, Callable
import math
from triton.language.semantic import TritonSemantic
from . import _core as ttgl
from ._layouts import AutoLayout, DistributedLayout, DistributedLinearLayout, SliceLayout, SharedLayout, CoalescedLayout, SharedLinearLayout
from triton._C.libtriton.gluon_ir import GluonOpBuilder, compute_tmem_reg_layout
from triton._C.libtriton import ir
from triton.compiler.code_generator import flatten_values_to_ir, unflatten_ir_values
```
No new imports needed — `call_extern` stays backend-agnostic. Access the hook via `self.builder.codegen_fns.get("infer_extern_call_result")`.

**Hook access pattern** — `codegen_fns` is stored on `self.builder` by `code_generator.py:312`:
```python
# code_generator.py line 312:
self.builder.codegen_fns = codegen_fns
```

**ScalarType → dtype mapping** (from `compiler.py:164-170` + `_core.py:39-41`):
```python
_scalar_to_dtype = {
    llvm.ScalarType.Fp32: ttgl.float32,
    llvm.ScalarType.Fp16: ttgl.float16,
    llvm.ScalarType.Bf16: ttgl.bfloat16,
    llvm.ScalarType.Int32: ttgl.int32,
    llvm.ScalarType.Int64: ttgl.int64,
}
```
This mapping goes in the **hook** (`compiler.py:InferExternCallResult.infer_result`), NOT in `_semantic.py`. The hook returns `(scalar, shape)` as a tuple, and the hook itself does the `ScalarType → scalar name` mapping. `_semantic.py` receives scalar names as strings (e.g. `"f32"`, `"i32"`).

**Core pattern — modified `call_extern`** (lines 250-279, modifications in **bold**):
```python
def call_extern(self, src_path, func, args, result_layouts, assert_no_conv=False, use_fast_math=False):
    _check(isinstance(src_path, str), lambda: f"expected 'src_path' to be a str but got {type(src_path)!r}")
    _check(isinstance(func, str), lambda: f"expected 'func' to be a str but got {type(func)!r}")
    for a in args:
        _check(isinstance(a, ttgl.tensor),
               lambda: f"all arguments must be tensors but got {type(a)!r}")
    _check(isinstance(result_layouts, list),
           lambda: f"result_layouts must be a list but got {type(result_layouts)!r}")

    # Backend-agnostic f64 guard — raise before building IR.
    _f64_err = "gl.call() does not support float64; full Fp64 support is out of scope (see FP64-01)"
    for a in args:
        if hasattr(a, 'dtype') and str(a.dtype) in ("fp64", "f64", "float64"):
            raise NotImplementedError(_f64_err)

    # ---- PHASE 2: Use CUDA-inferred dtype+shape at semantic time ----
    # Access the inference hook from codegen_fns (backend-agnostic — no NVIDIA imports).
    infer_hook = self.builder.codegen_fns.get("infer_extern_call_result")
    
    if infer_hook is not None:
        # Build arg_params for inference: dtype + shape + layout info from each arg.
        # The arg handles are not needed for dtype+shape inference — only their types.
        arg_params = []
        for a in args:
            arg_params.append({
                "dtype": str(a.dtype),
                "shape": list(a.shape),
                "layout": a.type.layout,  # may be AutoLayout at this point
            })
        # Call into the hook (backend-agnostic: receives back per-result (scalar_name, shape))
        inferred_results = infer_hook.infer_result(func, arg_params, use_fast_math)
        # inferred_results is a list of (scalar_name, shape) per result
        
        # Guard: result count mismatch
        _check(len(inferred_results) == len(result_layouts),
               lambda: f"gl.call(\"{func}\"): CUDA infers {len(inferred_results)} result(s) "
                       f"but {len(result_layouts)} result_layout(s) declared")
    else:
        inferred_results = None

    # Build result types
    result_types = []
    for i, lo in enumerate(result_layouts):
        if inferred_results is not None:
            scalar_name, result_shape = inferred_results[i]
            # Map scalar name to triton dtype
            _scalar_map = {"f32": ttgl.float32, "f16": ttgl.float16,
                           "bf16": ttgl.bfloat16, "i32": ttgl.int32, "i64": ttgl.int64}
            inferred_dtype = _scalar_map.get(scalar_name, ttgl.float32)
        else:
            # Fallback: infer from first_input (Phase 1 behavior, preserved for non-CUDA backends)
            first_input = args[0]
            inferred_dtype = first_input.dtype
            result_shape = _compute_result_shape(first_input.shape, lo)

        result_types.append(
            ttgl.distributed_type(inferred_dtype, result_shape, lo))

    result_ir_types = [rt.to_ir(self.builder) for rt in result_types]
    arg_handles = [a.handle for a in args]
    result_handles = self.builder.create_extern_call(
        str(src_path), func, arg_handles, result_ir_types,
        assert_no_conv, use_fast_math)

    results = [ttgl.tensor(h, rt) for h, rt in zip(result_handles, result_types)]
    return results[0] if len(results) == 1 else tuple(results)
```

**Key patterns to follow:**
- `_check(cond, lambda_fn, category=ValueError)` for all validations (line 13-15)
- `ttgl.distributed_type(dtype, shape, layout)` to construct result types (line 270)
- `self.builder.codegen_fns.get(...)` for backend-agnostic hook access
- No NVIDIA backend imports in this file (backend-agnostic rule, D-03)

**No change to:**
- `_compute_result_shape` (lines 18-28) — still used as fallback or when hook absent
- `to_linear_layout` (lines 427-440) — unchanged; AutoLayout passthrough (line 437-438) preserved

---

### 2. `third_party/nvidia/backend/compiler.py` — `InferExternCallResult.infer_result` (service/hook, request-response)

**Analog:** `compiler.py:228-252` (`compile_bitcode` method in same `InferExternCallResult` class)

**Existing class structure** (lines 190-252):
```python
class InferExternCallResult:

    def __init__(self, sm, resource_dir, include_paths):
        import triton._C.libtriton.llvm as _llvm
        self._sm = sm
        self._resource_dir = resource_dir
        self._include_paths = include_paths
        self._compilers = {}  # libpath -> SuspendedCudaCompiler
        self._llvm_ctx = None
        self._parse_count_before = _llvm.get_extern_cuda_parse_count()

    def create_and_suspend(self, source, llvm_context, libpath):
        import triton._C.libtriton.llvm as llvm
        if self._llvm_ctx is None:
            self._llvm_ctx = llvm_context
        compiler = llvm.SuspendedCudaCompiler(
            source, 3, self._sm, self._resource_dir, self._include_paths)
        compiler.parse(llvm_context, "cudamod")
        self._compilers[libpath] = compiler

    def infer_result(self, func, arg_params, use_fast_math):
        """Infer the CUDA return type for a function call. (Phase 2 consumer)"""
        raise NotImplementedError(
            "infer_result: return-type inference not available in Phase 1")

    def compile_bitcode(self, libpath, requests):
        compiler = self._compilers.get(libpath)
        if compiler is None:
            raise RuntimeError(...)
        ok, bitcode, error, results = compiler.compile_bitcode(requests)
        ...
```

**Core pattern — `infer_result` implementation** (replaces stub at line 220):

```python
def infer_result(self, func, arg_params, use_fast_math):
    """Infer the CUDA return type (scalar, shape) at semantic time.
    
    Uses PlaceholderLayout-based probing to determine dtype+shape
    without requiring concrete layouts. Returns a list of
    (scalar_name, shape) tuples, one per result.
    """
    import triton._C.libtriton.llvm as llvm
    
    _libpath = None
    for libpath, compiler in self._compilers.items():
        _libpath = libpath
        break
    
    if _libpath is None:
        raise RuntimeError(
            "infer_result: no suspended compilers available — "
            "make_ir must create a SuspendedCudaCompiler first")
    
    compiler = self._compilers[_libpath]
    
    # Build CudaFuncRequest — same pattern as _pre_compile_extern_calls (lines 684-702)
    req = llvm.CudaFuncRequest()
    req.symbol = func
    req.use_fast_math = use_fast_math
    
    param_types = []
    for ap in arg_params:
        tp = llvm.TensorParameter()
        tp.type = _scalar_type_for(ap["dtype"])
        tp.shape = ap["shape"]
        tp.layout_shape = ap["shape"]
        # PlaceholderLayout: pass empty bases — the inference-only path
        # uses these only to build Tensor<T,Shape,PlaceholderLayout> args.
        tp.reg_basis = []
        tp.lane_basis = []
        tp.warp_basis = []
        tp.n_warps = 0  # placeholder
        param_types.append(tp)
    req.param_types = param_types
    
    # Call inference-only C++ method (D-01): returns per-result (scalar, shape)
    ok, error, results = compiler.infer([req])
    if not ok:
        raise RuntimeError(
            f"infer_result: CUDA type inference failed for '{func}': {error}")
    
    _scalar_names = {
        llvm.ScalarType.Fp32: "f32", llvm.ScalarType.Fp16: "f16",
        llvm.ScalarType.Bf16: "bf16", llvm.ScalarType.Int32: "i32",
        llvm.ScalarType.Int64: "i64",
    }
    inferred = []
    for r in results:
        for tp in r.return_types:
            scalar_name = _scalar_names.get(tp.type, "f32")
            inferred.append((scalar_name, list(tp.shape)))
    
    if len(inferred) == 0:
        raise RuntimeError(
            f"infer_result: no return types inferred for '{func}'")
    return inferred
```

**Key patterns to follow:**
- `_scalar_type_for(dtype_str)` helper already exists at lines 657-664
- `CudaFuncRequest` construction at lines 686-702 (same `req.param_types = param_types`)
- `TensorParameter` field assignment at lines 693-701
- `_scalar_names` mapping at lines 166-170 (`_serialize_return_types`)
- Error raising pattern: `raise RuntimeError(f"...")` (lines 236-238, 745-746)

---

### 3. `python/src/clang_compiler.cc` — `CUDACompiler::inferReturnType` (service, request-response)

**Analog:** `clang_compiler.cc:809-936` (`compileBitcode` method — same class, same Phase 1 pattern for inference)

**Existing compileBitcode phases** (lines 809-936):
```cpp
// Phase 1: Type inference for all calls (lines 830-888)
// Phase 2: Codegen for all resolved functions (lines 892-922)
// Phase 3: Finalize — emit LLVM module (lines 924-936)
```

**Core pattern — `inferReturnType` (inference-only, no codegen):**

The new method copies Phase 1 (lines 830-888) of `compileBitcode` exactly — BuildTensor, LookupFunction, EvaluateFunctionReturnType — but stops before `InstantiationFunction` and `EmitFinalModule`.

```cpp
// Signature in clang_compiler.cc (NEW method, next to compileBitcode):
std::tuple<std::string, std::string, std::vector<CudaFuncResult>>
CUDACompiler::inferReturnTypes(
    const std::vector<CudaFuncRequest> &requests) {
  std::vector<CudaFuncResult> results;

  // Only Phase 1: Type inference (no codegen, no emit)
  for (auto &req : requests) {
    // Build clang QualType args from TensorParameter/ScalarType
    // (IDENTICAL to compileBitcode lines 832-851)
    llvm::SmallVector<clang::QualType, 4> argTypes(
        req.ParamTypes.size());

    for (size_t J = 0; J < req.ParamTypes.size(); J++) {
      if (auto *tp =
              std::get_if<TensorParameter>(&req.ParamTypes[J])) {
        argTypes[J] = this->BuildTensor(*tp);
      } else if (auto *st = std::get_if<ScalarType>(
                     &req.ParamTypes[J])) {
        clang::QualType Result;
        this->TaskQueue.emplace(
            [&, st](TensorTypeHelpers &helper,
                    CustomAstConsumer &) {
              Result = getQualTypeFromScalarType(helper.Builder.Ctx,
                                                  *st);
            });
        this->InvocationContext->SwitchTo(
            *this->CompileExecutionContext);
        argTypes[J] = Result;
      }
    }

    // LookupFunction + EvaluateFunctionReturnType
    // (IDENTICAL to compileBitcode lines 854-868)
    auto *FD =
        this->LookupFunction(req.Symbol, argTypes);
    if (!FD)
      return {"", "Function lookup failed: " + req.Symbol, {}};

    auto ret = this->EvaluateFunctionReturnType(FD);
    std::vector<TensorParameter> retParams;
    if (auto *tp = std::get_if<TensorParameter>(&ret)) {
      retParams.push_back(std::move(*tp));
    } else if (auto *tt = std::get_if<TupleType>(&ret)) {
      for (auto &elem : tt->Types) {
        if (auto *tp = std::get_if<TensorParameter>(&elem))
          retParams.push_back(std::move(*tp));
      }
    }

    // Build result (NOT the full CudaFuncResult — no mangled name, no bitcode)
    CudaFuncResult r;
    r.Symbol = req.Symbol;
    r.ReturnTypes = std::move(retParams);
    // r.MangledName stays empty — no codegen
    // r.ExtractorMangledNames stays empty — no extractors at inference time
    results.push_back(std::move(r));
  }

  // NO Phase 2 (codegen), NO Phase 3 (emit)
  return {"", "", results};
}
```

**Key patterns to follow:**
- Coroutine pattern: `TaskQueue.emplace(...)` + `InvocationContext->SwitchTo(...)` — every cross-context call MUST use this (lines 754-758, 784-788, etc.)
- `BuildTensor(tp)` for TensorParameter → QualType (line 838)
- `LookupFunction(Name, ArgTypes)` for overload resolution (line 854)
- `EvaluateFunctionReturnType(FD)` for return type inspection (line 859)
- Return type: `std::tuple<std::string, std::string, std::vector<CudaFuncResult>>` matching `compileBitcode` signature
- For single-return: `TensorParameter` variant (line 861); for tuple-return: `TupleType` with per-element `TensorParameter`s (lines 863-868)

**PlaceholderLayout integration (D-05/D-06):** The `BuildTensor` method (line 336 of header) already takes a `TensorParameter` and builds the clang type. When the bases are empty/zero (placeholder), it should build `Tensor<T, Shape, PlaceholderLayout>` instead of a concrete `Layout<REGS,LANES,WARPS>`. The inference-only path passes `reg_basis=[], lane_basis=[], warp_basis=[], n_warps=0` from the Python side. `BuildTensor` detects this and substitutes `PlaceholderLayout`.

---

### 4. `python/src/clang_compiler.h` — `inferReturnType` declaration (header)

**Analog:** `clang_compiler.h:348-349` (`compileBitcode` declaration)

**Core pattern — add declaration immediately after `compileBitcode`:**
```cpp
  // INFER-07: Split-path compile — runs inference+codegen+emit phases
  // on an already-parsed compiler (Parse must have been called first).
  std::tuple<std::string, std::string, std::vector<CudaFuncResult>>
  compileBitcode(const std::vector<CudaFuncRequest> &requests);

  // D-01: Inference-only — runs type inference (Phase 1 of compileBitcode)
  // on an already-parsed compiler without emitting LLVM bitcode.
  // Returns per-request TensorParameter results (no mangled names, no bitcode).
  std::tuple<std::string, std::string, std::vector<CudaFuncResult>>
  inferReturnTypes(const std::vector<CudaFuncRequest> &requests);
```

**Surrounding context to match** (lines 346-349):
```cpp
  // INFER-07: Split-path compile — runs inference+codegen+emit phases
  // on an already-parsed compiler (Parse must have been called first).
  std::tuple<std::string, std::string, std::vector<CudaFuncResult>>
  compileBitcode(const std::vector<CudaFuncRequest> &requests);
};
```

---

### 5. `python/src/llvm.cc` — `SuspendedCudaCompiler.infer` (Python binding)

**Analog:** `llvm.cc:1041-1066` (`compile_bitcode` binding on same py::class_)

**Existing binding pattern** (lines 1041-1066):
```cpp
py::class_<CUDACompiler>(m, "SuspendedCudaCompiler")
    .def(py::init<const std::string &, int, const std::string &,
                   const std::string &,
                   const std::vector<std::string> &>(),
         py::arg("source"), py::arg("opt_level"), py::arg("sm"),
         py::arg("resource_dir"), py::arg("include_paths"))
    .def("parse", ...)
    .def("compile_bitcode",
         [](CUDACompiler &compiler,
            const std::vector<CudaFuncRequest> &requests) -> py::tuple {
           auto [bitcode, error, results] =
               compiler.compileBitcode(requests);
           if (error.empty()) {
             py::list pyResults;
             for (auto &r : results) {
               py::list pyRetTypes;
               for (auto &tp : r.ReturnTypes)
                 pyRetTypes.append(py::cast(tp));
               py::list pyExtrNames;
               for (auto &n : r.ExtractorMangledNames)
                 pyExtrNames.append(py::str(n));
               pyResults.append(
                   py::make_tuple(r.Symbol, r.MangledName,
                                   pyRetTypes, pyExtrNames));
             }
             return py::make_tuple(true, py::bytes(bitcode),
                                    py::str(""), pyResults);
           } else {
             return py::make_tuple(false, py::none(),
                                    py::str(error), py::list());
           }
         },
         py::arg("requests"));
```

**Core pattern — add `infer` method after `compile_bitcode`:**

```cpp
      .def("infer",
           [](CUDACompiler &compiler,
              const std::vector<CudaFuncRequest> &requests) -> py::tuple {
             auto [bitcode, error, results] =
                 compiler.inferReturnTypes(requests);
             if (error.empty()) {
               py::list pyResults;
               for (auto &r : results) {
                 py::list pyRetTypes;
                 for (auto &tp : r.ReturnTypes)
                   pyRetTypes.append(py::cast(tp));
                 pyResults.append(
                     py::make_tuple(r.Symbol, py::str(""),
                                     pyRetTypes, py::list()));
               }
               return py::make_tuple(true, py::bytes(bitcode),
                                      py::str(""), pyResults);
             } else {
               return py::make_tuple(false, py::none(),
                                      py::str(error), py::list());
             }
           },
           py::arg("requests"));
```

**Key differences from `compile_bitcode` binding:**
- Calls `compiler.inferReturnTypes(requests)` instead of `compileBitcode(requests)`
- No `ExtractorMangledNames` — pass empty `py::list()` for that field
- `MangledName` is empty string — no codegen
- `bitcode` is empty — no emit
- Same return tuple structure: `(ok, bitcode, error, results)` — preserving the `compile_bitcode` convention for consistency

**Return type for Python consumer:** `results[i] = (symbol, "", [TensorParameter, ...], [])`. The Python side accesses `r.ReturnTypes` via `r.return_types` (line 1015) — each `TensorParameter` has `.type` (ScalarType enum), `.shape` (list of ints), `.layout_shape`, `.reg_basis`, `.lane_basis`, `.warp_basis`, `.n_warps`.

---

### 6. `python/test/gluon/tt_plugin.cu` — `PlaceholderLayout` + implicit conversion (library)

**Analog:** `tt_plugin.cu:65-78` (existing `Layout` template)

**Existing `Layout` pattern** (lines 65-78):
```cpp
    template<BasisGroup<N_REG_AXES> REGS, BasisGroup<N_LANE_AXES> LANES, BasisGroup<N_WARP_AXES> WARPS>
    struct Layout{
        template<int SLICE_DIM>
        using Sliced = Layout<REGS.sliceOut(SLICE_DIM), LANES.sliceOut(SLICE_DIM), WARPS.sliceOut(SLICE_DIM)>;

        static constexpr uint32_t NUM_WARPS = N_WARPS;
        static constexpr uint32_t REG_SIZE = 1u << N_REG_AXES;
        // ... static members ...
    };
```

**Existing `Tensor` template** (lines 81-84):
```cpp
template<typename T, typename TShape, typename TLayout>
struct Tensor{
    T data[TLayout::REG_SIZE];
};
```

**Core pattern — PlaceholderLayout + implicit conversion:**

```cpp
// PlaceholderLayout: empty marker type for dtype+shape-only inference.
// When used as Tensor<T,Shape,PlaceholderLayout>, it matches any
// Tensor<T,Shape,ConcreteLayout> via the implicit conversion below.
struct PlaceholderLayout {};

// Implicit conversion: Tensor<T,Shape,PlaceholderLayout> → Tensor<T,Shape,ConcreteLayout>
// for any concrete Layout<REGS,LANES,WARPS>.
// This enables the CUDA template deduction engine to match placeholder-probed
// args against real device function signatures, resolving T and Shape
// without needing concrete reg/lane/warp bases.
template<typename T, typename TShape, typename TLayout>
struct Tensor{
    T data[TLayout::REG_SIZE];
    
    // IMPLICIT CONVERSION from Placeholder-typed Tensor
    // (enables: Tensor<float,Shape<32>,PlaceholderLayout> → Tensor<float,Shape<32>,ConcreteLayout>)
    template<typename T2, typename TShape2>
    Tensor(const Tensor<T2, TShape2, PlaceholderLayout>& other) {
        // Static assert: T and TShape must match (deduced by compiler)
        static_assert(std::is_same_v<T, T2>, "dtype mismatch in PlaceholderLayout conversion");
        static_assert(std::is_same_v<TShape, TShape2>, "shape mismatch in PlaceholderLayout conversion");
        // Copy data registers from placeholder Tensor
        #pragma unroll TLayout::REG_SIZE
        for(uint32_t i = 0; i < TLayout::REG_SIZE; i++)
            data[i] = other.data[i];
    }
};
```

**Where to place:**
- `PlaceholderLayout` struct: after `Layout` (line 78), before `Tensor` (line 81)
- Implicit conversion constructor: inside `Tensor` struct (lines 81-84), after `T data[TLayout::REG_SIZE];` (line 83)

**Why this works (D-05/D-06):** When the clang compiler builds `Tensor<T, Shape<dims...>, PlaceholderLayout>` as the argument type and calls `LookupFunction`, the template deduction engine uses the **implicit conversion** to match against any `Tensor<T, Shape<dims...>, ConcreteLayout>` signature. Since dtype+shape are layout-independent in this framework, the deduction succeeds and yields the correct `T` and `Shape`. Only layout-bearing overloads (where multiple candidates differ only by layout) would be ambiguous; for dtype+shape, there is exactly one matching candidate per `(T, Shape)` pair.

---

## Shared Patterns

### Coroutine Context-Switch Pattern (clang_compiler.cc)
**Source:** `clang_compiler.cc:754-758, 784-788`
**Apply to:** `inferReturnTypes` (all TaskQueue calls)

Every cross-context operation on `CUDACompiler` uses this pattern:
```cpp
this->TaskQueue.emplace([&](TensorTypeHelpers &helper,
                             CustomAstConsumer &AstC) {
    /* ... clang Sema/CodeGen operations ... */
});
this->InvocationContext->SwitchTo(*this->CompileExecutionContext);
```
Never call `clang::Sema` or `clang::CodeGen` methods directly — always enqueue and switch.

### ScalarType Mapping (Python → C++ ↔ Python)
**Sources:** `compiler.py:648-654` (dtype→ScalarType), `compiler.py:166-170` (ScalarType→name), `llvm.cc:946-953` (enum definition)

```
Python dtype string → ScalarType enum:
  "f32"/"fp32" → Fp32, "f16"/"fp16" → Fp16, "bf16" → Bf16,
  "i32"/"s32" → Int32, "i64"/"s64" → Int64

ScalarType enum → Python scalar name:
  Fp32 → "f32", Fp16 → "f16", Bf16 → "bf16",
  Int32 → "i32", Int64 → "i64"

ScalarType enum → clang QualType:
  Fp32 → Ctx.FloatTy, Fp16 → Ctx.HalfTy,
  Int32 → Ctx.IntTy, Int64 → Ctx.LongLongTy
```

### Backend-Agnostic Hook Pattern
**Source:** `_runtime.py:91-99`, `compiler.py:340-342`

Hook creation: `get_codegen_implementation` adds the hook to `codegen_fns` dict.
Hook access in `_semantic.py`: `self.builder.codegen_fns.get("infer_extern_call_result")`.
Hook access in `_runtime.py` (make_ir): `codegen_fns.get("infer_extern_call_result")`.
Hook access in `compiler.py` (_pre_compile_extern_calls): `getattr(self, '_infer_hook', None)`.

### MLIR Op Construction Pattern (call_extern)
**Source:** `_semantic.py:250-279`

1. Build `distributed_type(element_ty, shape, layout)` for each result
2. Convert to IR via `.to_ir(self.builder)`
3. Call `self.builder.create_extern_call(src_path, func, arg_handles, result_ir_types, assert_no_conv, use_fast_math)`
4. Wrap results in `ttgl.tensor(handle, distributed_type)`
5. Return single tensor or tuple

### Error Handling Patterns

**Python validation** (`_semantic.py:13-15`):
```python
def _check(cond: bool, msg_fn: Callable[[], str], category=ValueError):
    if not cond:
        raise category(msg_fn())
```

**C++ error return** (`compileBitcode` convention, lines 820-822):
```cpp
return {"", "error message describing the failure", {}};
```
Return tuple is `{bitcode (empty=error), error_message (empty=success), results}`.

**Python RuntimeError** (`compiler.py`):
```python
raise RuntimeError(f"In-process CUDA compilation failed for {libpath}: {error}")
```

### Test Pattern (existing E2E tests)
**Source:** `python/test/gluon/test_extern_call.py`

All four existing tests use:
```python
@gluon.jit
def kernel(x_ptr, ... , out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([...], [...], [...], [...])
    idx = gl.arange(0, N, layout=layout)
    vals = gl.load(ptr + idx)
    result = gl.call("python/test/gluon/tt_plugin.cu", "func_name", vals, result_layout=layout)
    gl.store(out_ptr + idx, result)

# Then run: kernel[(1,)](x, out, num_warps=1)
# Then: torch.testing.assert_close(out, ref)
```
**Phase 2 regression requirement:** These 4 tests MUST pass unchanged. The C++ patch step (`tritonPatchExternCallResultTypes`) continues to handle layout-only changes — dtype+shape are now set correctly at semantic time.

---

## Metadata

**Analog search scope:**
- `python/triton/experimental/gluon/language/_semantic.py`
- `third_party/nvidia/backend/compiler.py`
- `python/src/clang_compiler.cc` / `.h`
- `python/src/llvm.cc`
- `python/test/gluon/tt_plugin.cu`
- `python/test/gluon/test_extern_call.py`

**Files scanned:** 12+
**Pattern extraction date:** 2026-07-11
