# Working on Triton

## Build
- Initialize: `pip install -r python/requirements.txt && pip install -e .` (or `make dev-install`)
- For C++/compiler changes only, build with `make` (runs `ninja -C $BUILD_DIR`). Do NOT run `make` if you only changed Python code or `python/triton_kernels`.
- Build dir: `BUILD_DIR := $(shell PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`. Pattern: `build/cmake.<plat>-<py_impl>-<py_ver>/`. Override with `TRITON_BUILD_DIR`.
- Build a single target: `ninja -C $BUILD_DIR triton-opt` (or `make triton-opt`).

## Testing
- `make test-lit` — lit tests (no GPU needed, from build dir).
- `make test-cpp` — C++ unit tests (no GPU needed).
- `make test-unit` — Python GPU tests (pytest, parallel by default via `-n 8`).
- `make test-regression`, `make test-gluon`, `make test-proton`, `make test-interpret` — other suites.
- Run lit: `cd $BUILD_DIR && ninja triton-opt && lit -v test/<path>.mlir`
- Run single pytest: `pytest -s --tb=short file.py::test_name`. Pytest `addopts = --tb=short` is in `pytest.ini`.
- GPU-only tests go in `python/test/unit/` or `python/test/gluon/`. Name them `test_<feature>_<condition>`.

## Compiler Pipeline (CUDABackend)
The compiler lowering runs as an ordered dict of stage extensions, each a function `compile_ir(mod, metadata) -> mod`:

1. **ttir** — Triton IR: inlining, canonicalization, CSE, loop unroll. Python AST → MLIR `tt` dialect.
2. **ttgir** — TritonGPU IR: `tt` → `ttg` dialect conversion, coalescing, matmul acceleration, layout conversion, software pipelining, warp specialization, etc.
3. **llir** — LLVM IR: `ttg` → LLVM dialect conversion (`ConvertTritonGPUToLLVM` pass), MLIR → `llvm::Module` translation, **extern libs linking**, LLVM optimization (O3 + BreakStructPhiNodes), NVVM→LLVM conversion.
4. **ptx** — LLVM → PTX assembly via `TargetMachine::addPassesToEmitFile`.
5. **cubin** — PTX → CUDA binary via `ptxas` (`make_cubin` calls `ptxas` + driver `load_binary`).

Key orchestration file: `third_party/nvidia/backend/compiler.py` (class `CUDABackend`).

## Extern Libs / External CUDA C++ Interop
- External device functions (e.g. libdevice) are linked as LLVM bitcode (`.bc`) at the **llir** stage.
- User passes `extern_libs={'name': '/path/to/lib.bc'}` in kernel launch kwargs.
- `make_llir()` checks `has_extern_deps()` (any extern linkage symbols), then calls `llvm.link_extern_libs(mod, paths)` which uses `llvm::Linker::linkInModule` with `LinkOnlyNeeded`.
- Linked-in extern functions are set to `InternalLinkage` to avoid being mistaken for kernel entry points.
- C++ implementation of linking: `python/src/llvm.cc:916` (`link_extern_libs`).

### Adding a new extern function call
Python-side pattern (see `third_party/nvidia/language/cuda/libdevice.py`):
1. Define a `@core.extern` (or `@builtin`) function that calls `core.extern_elementwise(lib_name, lib_path, [args], type_dict, is_pure, _semantic)`.
2. This emits a `tt.extern_elementwise` op with `symbol="__my_func"`.
3. At LLVM lowering (`lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp:183`), it becomes an `LLVM::CallOp` to `@__my_func`.
4. The bitcode containing `__my_func` is linked at the llir stage.

### Custom pipeline stages (plugins)
- Inject via `knobs.runtime.add_stages_inspection_hook`. The hook receives `stages` dict and can wrap/replace stage functions.
- Example: `python/test/unit/plugins/custom_stages.py` — replaces the `"ttir"` stage with a wrapper that runs a custom MLIR pass.
- For custom MLIR ops: `TRITON_EXT_ENABLED=1` + `TRITON_PLUGIN_PATHS=path/to/libPlugin.so` to load an MLIR dialect plugin. Define ops with `@builtin` + `builder.create_*` bindings.
- See `python/test/unit/plugins/custom_ops.py` for an example custom op defined via `@builtin`.

## LinearLayout System
- Mathematically: layouts are linear functions over GF(2) mapping hardware locations (`register`, `lane`, `warp`, `block`) → logical tensor indices (`dim0`, `dim1`, ...).
- Core class: `include/triton/Tools/LinearLayout.h` / `lib/Tools/LinearLayout.cpp` (uses `f2reduce` for GF(2) linear algebra).
- Conversion from `ttg` encoding attributes to LinearLayout: `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` (e.g. `blockedToLinearLayout`, `mmaToLinearLayout`, `swizzledSharedToLinearLayout`).
- Codegen applies layouts: `lib/Conversion/TritonGPUToLLVM/Utility.cpp` `applyLinearLayout()` — emits LLVM IR that computes thread/lane/warp indices from layout bases.
- Python bindings: `python/src/linear_layout.cc` → exposed as `triton.tools.LinearLayout`.

## Key C++ Source Locations
| Purpose | Path |
|---------|------|
| Triton dialect ODS | `include/triton/Dialect/Triton/IR/TritonOps.td` |
| TritonGPU dialect ODS | `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` |
| Triton→TritonGPU conversion | `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` |
| TritonGPU→LLVM lowering (base) | `lib/Conversion/TritonGPUToLLVM/` |
| TritonGPU→LLVM (NVIDIA-specific) | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp` |
| ExternElementwiseOp lowering | `lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp:183` |
| FuncOp/CallOp lowering | `lib/Conversion/TritonGPUToLLVM/FuncOpToLLVM.cpp`, `ControlFlowOpToLLVM.cpp` |
| Python IR builder bindings | `python/src/ir.cc` |
| Python LLVM tool bindings | `python/src/llvm.cc` |
| NVIDIA Python compiler | `third_party/nvidia/backend/compiler.py` |
| Python semantic layer | `python/triton/language/semantic.py` |
| Language core (builtins) | `python/triton/language/core.py` |
| Code generator (AST→IR) | `python/triton/compiler/code_generator.py` |
| Top-level compiler orchestration | `python/triton/compiler/compiler.py` |

## Reproducing Compiler Crashes
Compiler crashes sometimes print an MLIR reproducer. Save the full MLIR + `{-# ... #-}` metadata to `/tmp/<file>.mlir`, then run:
```
triton-opt /tmp/<file>.mlir --run-reproducer
```

## Linting
- `ruff --fix` for Python under `python/`, `third_party/{amd,nvidia,proton}/`, `test/`
- `yapf -p -i` for Python formatting
- `clang-format` for C++/CUDA
- `mypy` for type checking
- Config: `.pre-commit-config.yaml`, `pyproject.toml` (ruff/yapf/mypy config), `.clang-format`
