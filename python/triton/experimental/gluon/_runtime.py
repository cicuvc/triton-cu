from __future__ import annotations
from triton.compiler.compiler import ASTSource
from triton.backends.compiler import Language
from triton.runtime.jit import JITFunction, constexpr_function
from typing import TypeVar, Optional, Callable, Iterable, Union, overload
from triton._C.libtriton import ir

T = TypeVar("T")

__all__ = ["GluonJITFunction", "constexpr_function", "jit"]


class GluonASTSource(ASTSource):

    def __init__(self, fn, signature, constexprs=None, attrs=None) -> None:
        super().__init__(fn, signature, constexprs, attrs)
        self.language = Language.GLUON
        self.ext = "ttgir"

    def parse_options(self):
        import re
        from pathlib import Path

        # self.fn is the JITFunction instance; use .raw_src to get source.
        # Scan for gl.call("file.cu") patterns and add .cu paths to
        # extern_libs so file_hash() includes them in the cache key.
        extern_hashes = {}
        try:
            if hasattr(self.fn, 'raw_src'):
                source = ''.join(self.fn.raw_src)
            else:
                return {}
            for m in re.finditer(r'gl\s*\.\s*call\s*\(\s*["\']([^"\']+\.cu)["\']', source):
                cu_path = Path(m.group(1))
                if not cu_path.is_absolute():
                    cu_path = Path.cwd() / cu_path
                cu_path = cu_path.resolve()
                if cu_path.exists():
                    extern_hashes[f"__extern_call_src_{cu_path.stem}"] = str(cu_path)
        except Exception:
            pass

        if extern_hashes:
            return {"extern_libs": extern_hashes}
        return {}

    def make_ir(self, target, options, codegen_fns, module_map, context):
        from triton.compiler.compiler import make_backend
        from triton.compiler.code_generator import ast_to_ttir

        builder = ir.builder(context)
        module = builder.create_module()

        # Assign module attributes eagerly, as they are needed to verify layouts
        backend = make_backend(target)
        target = backend.get_target_name(options)

        module.set_attr("ttg.target", builder.get_string_attr(target))
        module.set_attr("ttg.num-warps", builder.get_int32_attr(options.num_warps))
        module.set_attr("ttg.num-ctas", builder.get_int32_attr(options.num_ctas))
        module.set_attr("ttg.threads-per-warp", builder.get_int32_attr(options.warp_size))

        is_cuda = options.backend_name == "cuda"
        if is_cuda and options.maxnreg is not None:
            module.set_attr("ttg.maxnreg", builder.get_int32_attr(options.maxnreg))

        # D-05/D-06: Pre-scan kernel source for gl.call(".cu") patterns.
        # Create+suspend a CUDACompiler for each distinct .cu file so
        # _pre_compile_extern_calls can resume them at the llir stage.
        if is_cuda:
            from pathlib import Path
            import re as _re
            _cu_paths = set()
            try:
                if hasattr(self.fn, 'raw_src'):
                    _source = ''.join(self.fn.raw_src)
                else:
                    _source = ""
                for _m in _re.finditer(
                        r'gl\s*\.\s*call\s*\(\s*["\']([^"\']+\.cu)["\']',
                        _source):
                    _cu_path = Path(_m.group(1))
                    if not _cu_path.is_absolute():
                        _cu_path = Path.cwd() / _cu_path
                    _cu_path = _cu_path.resolve()
                    if _cu_path.exists():
                        _cu_paths.add(str(_cu_path))
            except Exception:
                _cu_paths = set()

            _hook = codegen_fns.get("infer_extern_call_result")
            if _hook is not None:
                import triton._C.libtriton.llvm as _llvm
                _llvm.init_targets()
                _llvm_ctx = _llvm.context()
                for _cu_path in _cu_paths:
                    with open(_cu_path) as _f:
                        _cu_source = _f.read()
                    _hook.create_and_suspend(_cu_source, _llvm_ctx, _cu_path)

        module = ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                             module_map=module_map, module=module)
        return module


class GluonJITFunction(JITFunction[T]):

    def create_binder(self):
        result = super().create_binder()
        self.ASTSource = GluonASTSource
        return result

    def is_gluon(self):
        return True


@overload
def jit(fn: T) -> GluonJITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], GluonJITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[GluonJITFunction[T], Callable[[T], JITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        return GluonJITFunction(
            fn,
            version=version,
            do_not_specialize=do_not_specialize,
            do_not_specialize_on_alignment=do_not_specialize_on_alignment,
            debug=debug,
            noinline=noinline,
            repr=repr,
            launch_metadata=launch_metadata,
        )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
