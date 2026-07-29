# Working on triton-cu

A fork of [Triton](https://github.com/triton-lang/triton) adding in-process CUDA C++ interop via `gl.call()`.  
Remote: `git@github.com:cicuvc/triton-cu.git`

## Build
- **DO NOT run `pip install -e .`** — it overwrites the venv's standard triton. Use `PYTHONPATH` for local dev.
- **Use a self-compiled LLVM** via `-DLLVM_SYSPATH=...`. Set `LLVM_SYSPATH` env var to your LLVM install prefix (e.g. `/path/to/llvm/install`). `build.sh` checks this. Do NOT use Triton's default precompiled LLVM (symbol mismatch).
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
- **E2E test**: `pytest python/test/gluon/test_extern_call.py` (tests `gl.call` with in-process CUDA compilation).

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
_core.py:call() → _semantic.py:call_extern()
  Builds result_types from first_input.dtype/shape + user's result_layout
  Creates ttg.extern_call MLIR op (symbol=func, libpath=src_path)
  │
  ├──► _pre_compile_extern_calls() — before pm.run (compiler.py:491)
  │    extract_extern_call_specs() from MLIR ops → JSON (clang_compiler.cc:756)
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
| `python/src/gluon_ir.cc` | GluonOpBuilder: `create_extern_call`, `to_linear_layout`, `layoutToGluon` |
| `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` | `{[N x scalar]}` return type lowering, by-pointer arg convention |
| `third_party/nvidia/backend/compiler.py` | `_pre_compile_extern_calls()`, bitcode linking in `make_llir()` |
| `python/triton/experimental/gluon/language/_core.py` | `gl.call()` user-facing API (line 774) |
| `python/triton/experimental/gluon/language/_semantic.py` | `call_extern()` — builds result IR types from first arg (line 237-268) |
| `python/triton/experimental/gluon/language/_layouts.py` | DistributedLayout, BlockedLayout, DistributedLinearLayout Python types |
| `python/test/gluon/tt_plugin.cu` | CUDA C++ device library (Shape, BasisGroup, TensorLayout::Layout, Tensor templates) |
| `CMakeLists.txt` | Permanent Clang lib linkage, `-fno-rtti` for clang_compiler.cc, LLVMMIRParser |

### Important Details
- **Same LLVMContext for clone**: parsed bitcode uses Triton's context (not tmpCtx) to avoid metadata leaks.
- **Ret-type fix**: named `%struct.Tensor` from clone has different `Type*` than literal `{[16 x float]}` even in same LLVMContext. alloca+store+load launders the type.
- **Callee remapping**: intrinsic declarations (e.g. `llvm.lifetime.start`) are not auto-created in dstMod; must explicitly `Function::Create` them.
- **No wrappers**: C++ references become `ptr` params in LLVM IR. Lowering uses alloca+store+ptr matching clang's convention.
- **Memory effects**: `TTG_ExternCallOp` implements `MemoryEffectOpInterface` — each memdesc operand gets conservative `MemRead+MemWrite<SharedMemory>` effects (the C++ `SharedTensor&` signature does not distinguish). Membar therefore inserts barriers around `gl.call` automatically; user-written `gl.barrier()` remains compatible (dedup via sync).
- **No CUDA-internal sync**: device functions must NOT call `__syncthreads`, `__barrier_sync`, `__threadfence*` — once inlined into the kernel, MLIR cannot model such a synchronization point (and it can deadlock under warp specialization). `CUDACompiler::CheckForbiddenSync` rejects them at instantiation time, including through same-TU callees (uninstantiated template callees are a known blind spot).
- **fp16 element type is `_Float16`** (`Ctx.Float16Ty`), NOT `__fp16`: `__fp16` is storage-only in clang (illegal as function return/parameter type) and breaks `shared_accessor<T>` instantiation. Same `half` IR type either way.

### Shared-memory device library (tt_plugin.cu)
- `shared_accessor<T>` — scalar ld/st.shared at b8/b16/b32/b64 by `sizeof(T)`.
- `shared_load_vector` / `shared_store_vector` — 4/8/16-byte vector accesses
  (b32/v2.b32/v4.b32); `SharedTensor::loadVector<N>/storeVector<N>` add
  alignment `__trap` + debug contiguity assert.
- `SharedTensor::flattenCoords` / `byteOffsetOf` — constexpr index math shared
  by `operator()`, vector access, ldmatrix and cp.async paths.
- `ldmatrix_x4_b16(shm, fragBase, out[4])` — 16x16 b16 tile load; lane
  mapping and fragment layout documented at the definition.
- `cp_async_ca` / `cp_async_cg_16` / `cp_async_commit` / `cp_async_wait<N>`,
  plus `SharedTensor::cpAsyncFromGlobal<N>(gsrc, flatBase)`. Global pointers
  reach device functions as **i64 scalars** (`x.data_ptr()` via constexpr →
  `gl.call` scalar arg; int scalars beyond i32 range map to i64).
- **dtype chain**: `ScalarType` has `Int8` (SignedCharTy); bf16 maps to
  `Ctx.BFloat16Ty` for inference only — full bf16 E2E (cuda_bf16.h
  `__nv_bfloat16` type identity) and fp8 dtype mapping are still TODO.

### Direction: NVMMA / TMA / wgmma / tcgen05 (not implemented yet)
- `extractExternCallSpecs` currently flattens NVMMAShared encodings to raw
  offset bases via `nvmmaSharedToLinearLayout` — the swizzle-atom parameters
  (swizzle_byte_width, transposed, element_bit_width) are lost, so wgmma
  descriptors / TMA swizzle modes cannot be reconstructed on the CUDA side.
- Plan: recover `NVMMASharedLayout` from the raw linear layout (swizzle-atom
  recognition) rather than adding an encoding-kind channel. Linear-layout
  algebra (evaluation, row extraction, F2 matrix left-inverse) should be
  computed **in the compiler and injected** ("open a hole" in codegen), not
  reimplemented in CUDA C++ templates.
- tcgen05 introduces TMEM (`TensorMemoryEncodingAttr`) — a separate storage
  tier outside MemDescType/shared; needs its own spec channel later.

### Pure-Basis Layout Design (register tensors)
A distributed layout is a **pure linear map** — fully described by its basis
rows; it does NOT own a shape. The shape belongs solely to the `Tensor`.
This mirrors MLIR (`RankedTensorType{shape, encoding}`) and the shared-memory
side (`SharedLinearLayout` + standalone `OffsetBases`/`BlockBases` carriers).

- `BasisGroup<RANK, N_BASES>` — top-level NTTP carrier (evaluate/collectRow/sliceOut).
- `TensorLayout` — non-template shell; `TensorLayout::Layout<REGS, LANES, WARPS>`
  takes the three groups as `auto` NTTPs. `REG_SIZE`/`NUM_WARPS`/`N_*_AXES`
  derive from basis row counts, never from a shape (no `ffs` formula).
- `Tensor<T, Shape, Layout>` — static_asserts rank match + per-row basis
  bounds vs. shape dims (exact for power-of-two dims via XOR closure).
  `PlaceholderLayout` probes skip checks via the `ConcreteLayout` concept.
- Non-bijective (broadcast/sliced) layouts are expressible: `Sliced<D>`
  zeroes a dim's components, keeps row counts → duplicates materialize in
  registers, matching Triton broadcast semantics (see `add_bias` test).
- `LayoutInfo` (clang_compiler.h) = three flat basis vectors only — no
  `LayoutShape`, no `N_WARPS`. The basis component count always equals the
  tensor rank; each group's row count = `vec.size() / rank`.

### Current Limitation: Return Type Layout
`_semantic.py:246-252` infers `gl.call()` return layout entirely from `first_input.dtype` + `first_input.shape` — the user must manually supply the correct `result_layout=`. This is wrong for functions where the return type's element type, shape, or layout differs from the first argument (e.g., `add_bias` in `tt_plugin.cu:351` narrows the bias tensor shape to `[1, TILE_COLS]`).

## Return Type Inference (in-progress)
Goal: perform proper C++ overload resolution + template argument deduction + return type inspection in clang Sema to determine `gl.call()` return layout automatically.

### Reference Implementation
**`/home/cicuvc/cs/project/nks/lab/cu_compiler_v2.cpp` / `.h`** — standalone proof-of-concept with the full inference pipeline. Key abstractions to integrate:

| Abstraction | What it provides |
|-------------|-----------------|
| `TypeBuilder` | `TensorParameter` user data → clang AST types (Shape, Layout, Tensor) |
| `TypeInspector` | Reverse: clang AST `Tensor<T,S,L>` → `TensorParameter` (scalar type, shape dims, layout bases) |
| `FunctionResolver` | `LookupFunction()` with proper Sema `DeduceTemplateArguments` + `getMoreSpecializedTemplate` overload resolution |
| `CUDACompiler::EvaluateFunctionReturnType()` | Calls `TypeInspector::DispatchTypeParsing(FD->getCallResultType())` to get the actual return `TensorParameter` |

### Integration Plan
1. **Add `TypeInspector`** to `clang_compiler.cc` — parse clang `ClassTemplateSpecializationDecl` back to `TensorParameter` (scalar type, shape dims, layout bases).
2. **Separate `FunctionResolver`** from `TensorTypeHelpers::InstantiateFunction()` — currently it does lookup + instantiation inline; split so we can evaluate the return type *after* resolution but before codegen.
3. **Call `EvaluateFunctionReturnType()`** in `CustomAstConsumer::HandleTranslationUnit()` after `InstantiateFunction()` and store the result on the instantiation struct.
4. **Plumb return `TensorParameter` back** through the Python bindings (`compile_cuda_to_module` result tuple) so `_pre_compile_extern_calls()` can use it.
5. **Replace** `_semantic.py:246-252` with the inferred return layout instead of inferring from `first_input`.
6. **Convert** inferred `TensorParameter` (shape + bases) back to Triton's `DistributedLayout` via the `layoutToGluon` / `toLinearLayout` reverse path for constructing `result_layout` automatically.

### Layout Round-Trip (MLIR ↔ clang AST)
```
MLIR encoding (ttg.extern_call input operands)
  │  extractExternCallSpecs() — toLinearLayout(shape, encoding)
  ▼
TensorParameter {ScalarType, Shape, RegBasis, LaneBasis, WarpBasis}
  │  CustomAstConsumer — BuildBasisGroup → BuildLayout → BuildTensor
  ▼
clang AST: Tensor<{scalar}, Shape<dims...>, TensorLayout::Layout<REGS, LANES, WARPS>>
  │  FunctionResolver::LookupFunction + Sema template deduction
  ▼
clang FunctionDecl (resolved + instantiated)
  │  EvaluateFunctionReturnType → TypeInspector::DispatchTypeParsing
  ▼
TensorParameter (return type) → needs conversion back to MLIR/DistributedLayout
  └── layoutToGluon() + toLinearLayout reverse path
```

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
| Gluon Python API | `python/triton/experimental/gluon/language/_core.py` (gl.call), `_semantic.py` (call_extern), `_layouts.py` (DistributedLayout etc.) |
| Gluon runtime (cache key) | `python/triton/experimental/gluon/_runtime.py` (scans for gl.call patterns, adds .cu path to cache key) |
| layoutToGluon reverse path | `python/src/gluon_ir.cc:191-280` (MLIR encoding → Python DistributedLayout) |
| LinearLayout↔Encoding conversion | `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` (toLinearLayout, combineCtaCgaWithShape) |

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