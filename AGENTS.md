# Working on triton-cu

A fork of [Triton](https://github.com/triton-lang/triton) adding in-process CUDA C++ interop via `gl.call()`.  
Remote: `git@github.com:cicuvc/triton-cu.git`

## Build
- **DO NOT run `pip install -e .`** — it overwrites the venv's standard triton. Use `PYTHONPATH` for local dev.
- **Use our self-compiled LLVM** at `/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/llvm-data/install` via `-DLLVM_SYSPATH=...`. Do NOT use Triton's default precompiled LLVM (symbol mismatch).
- **Use clang as compiler** (`CC=clang CXX=clang++`). Our LLVM was built with clang; gcc is too slow / memory-hungry for this project.
- **Canonical build**: `bash build.sh` (handles cmake + ninja + lld).
  - Clang libs (CodeGen, Frontend, Driver, Sema, AST, etc.) and LLVMMIRParser are **permanently** in `CMakeLists.txt`.
  - `clang_compiler.cc` is compiled with `-fno-rtti` (Clang libs are built without RTTI).
- Build output: `build/libtriton.so`. Copy to `python/triton/_C/libtriton.so` for local dev.
- Run with `PYTHONPATH` set to local source tree so our modified Python files take priority over the venv install:
  ```
  PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python3 ...
  ```

## Testing
- `make test-lit` — lit tests (no GPU needed, from build dir).
- `make test-unit` — Python GPU tests (pytest, parallel by default via `-n 8`).
- `make test-regression`, `make test-gluon`, `make test-proton`, `make test-interpret` — other suites.
- Run lit: `cd $BUILD_DIR && ninja triton-opt && lit -v test/<path>.mlir`
- Run single test: `pytest -s --tb=short python/test/unit/language/test_core.py::test_name`
- **E2E test**: `python3 gluon_vectorized_add_test.py` (tests `gl.call` with in-process CUDA compilation).

## Compiler Pipeline (CUDABackend)
The compiler lowering runs as an ordered dict of stage extensions, each a function `compile_ir(mod, metadata) -> mod`:

1. **ttir** — Triton IR: inlining, canonicalization, CSE, loop unroll. Python AST → MLIR `tt` dialect.
2. **ttgir** — TritonGPU IR: `tt` → `ttg` dialect conversion, coalescing, matmul acceleration, layout conversion, software pipelining, warp specialization, etc.
3. **llir** — LLVM IR: `ttg` → LLVM dialect conversion (`ConvertTritonGPUToLLVM` pass), MLIR → `llvm::Module` translation,
   **inline extern CUDA functions** (link_cuda_bitcode), LLVM optimization (O3), NVVM→LLVM conversion.
4. **ptx** — LLVM → PTX assembly via `TargetMachine::addPassesToEmitFile`.
5. **cubin** — PTX → CUDA binary via `ptxas`.

Key orchestration file: `third_party/nvidia/backend/compiler.py` (class `CUDABackend`).

## Extern CUDA C++ Interop (`gl.call()`)
In-process CUDA template instantiation via clang CodeGen at JIT time.

### Pipeline
```
Triton/Gluon kernel with gl.call()
  │
  ▼
MLIR tt.call_external op (extern_elementwise)
  │
  ├──► _pre_compile_extern_calls() — before pm.run
  │    extract_extern_call_specs() from MLIR ops
  │    compile_cuda_to_module() — clang CompilerInvocation
  │      TensorTypeHelpers: instantiate template device functions
  │      CustomAstConsumer + CustomFEAction: CreateLLVMCodeGen
  │      Returns bitcode bytes + mangled function names
  │    Store mangled names as module attribute
  │
  ▼
ttgir → llir lowering (ExternCallOpToLLVM.cpp)
  Build clang-compatible {[N x scalar]} struct types
  Args passed by pointer (alloca + store + ptr) matching C++ ref convention
  Post-call: extractvalue/insertvalue back to MLIR's flat {scalar x N} structs
  │
  ▼
llvm.to_module → link_cuda_bitcode (CloneFunctionInto)
  Same LLVMContext parsing
  CloneFunctionInto with DifferentModule flag
  Callee remapping (intrinsic decls created in dstMod)
  Ret-type fix: alloca + store + load launders named→literal struct Type*
  DISubprogram fix (strip debug)
  alwaysinline attribute → O3 inlines fully
```

### Key Files
| File | Role |
|------|------|
| `python/src/clang_compiler.cc` / `.h` | CUDACompiler, TensorTypeHelpers, CustomAstConsumer, extractExternCallSpecs, compileCudaToModule, linkBitcodeToModule |
| `python/src/llvm.cc` | Python bindings: ScalarType, TensorParameter, DeviceFunctionInstantiation, extract_extern_call_specs, compile_cuda_to_module, link_cuda_bitcode |
| `python/src/ir.cc` | `set_str_attr` on Operation/ModuleOp |
| `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` | `{[N x scalar]}` return type lowering, by-pointer arg convention |
| `third_party/nvidia/backend/compiler.py` | `_pre_compile_extern_calls()`, bitcode linking in `make_llir()` |
| `tt_plugin.cu` | CUDA C++ device library (Shape, TensorLayout, Layout, Tensor templates) |
| `CMakeLists.txt` | Permanent Clang lib linkage, `-fno-rtti` for clang_compiler.cc, LLVMMIRParser |

### Important Details
- **Same LLVMContext for clone**: parsed bitcode uses Triton's context (not tmpCtx) to avoid metadata leaks.
- **Ret-type fix**: named `%struct.Tensor` from clone has different `Type*` than literal `{[16 x float]}` even in same LLVMContext. alloca+store+load launders the type.
- **Callee remapping**: intrinsic declarations (e.g. `llvm.lifetime.start`) are not auto-created in dstMod; must explicitly `Function::Create` them.
- **No wrappers**: C++ references become `ptr` params in LLVM IR. Lowering uses alloca+store+ptr matching clang's convention.

## Key C++ Source Locations
| Purpose | Path |
|---------|------|
| Triton dialect ODS | `include/triton/Dialect/Triton/IR/TritonOps.td` |
| TritonGPU dialect ODS | `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` |
| Triton→TritonGPU conversion | `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` |
| TritonGPU→LLVM lowering (base) | `lib/Conversion/TritonGPUToLLVM/` |
| TritonGPU→LLVM (NVIDIA-specific) | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp` |
| Extern call lowering | `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` |
| clang CUDA compiler | `python/src/clang_compiler.cc` / `python/src/clang_compiler.h` |
| Python bindings (CUDA interop) | `python/src/llvm.cc` (extract/cuda/link), `python/src/ir.cc` (set_str_attr) |
| NVIDIA Python compiler | `third_party/nvidia/backend/compiler.py` (class `CUDABackend`) |
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
- `ruff --fix` for Python under `python/`, `third_party/nvidia/`, `third_party/proton/`, `test/`
- `yapf -p -i` for Python formatting
- `clang-format` for C++/CUDA
- `mypy` for type checking
- Config: `.pre-commit-config.yaml`, `pyproject.toml` (ruff/yapf/mypy config), `.clang-format`