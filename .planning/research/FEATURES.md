# Feature Landscape: Shared Memory Interop for `gl.call()`

**Domain:** Compiler internals — CUDA/MLIR/Clang interop (NOT a product feature list)
**Researched:** 2026-07-12

## Table Stakes

Features that must exist for the milestone to be complete. Missing → milestone fails.

| Feature | Why Expected | Complexity | Integration Points |
|---------|--------------|------------|--------------------|
| `extern_call` op accepts `MemDescType` operands | Without this, shared-memory args can't be represented in MLIR | Med | `TritonGPUOps.td:803` (ODS change), `ExternCallOpToLLVM.cpp:111` (lowering dispatch) |
| `SharedLinearLayout` C++ device template | The C++ callee needs a concrete type to match against | Med | `tt_plugin.cu` (new template), `clang_compiler.cc` (TypeBuilder integration) |
| `SharedTensor<T,Shape,SharedLinearLayout>&` C++ device template | The C++ reference type the callee receives | Med | `tt_plugin.cu` (modeled on existing `Tensor`), `clang_compiler.h:224` (TypeBuilder extension) |
| `TypeBuilder::BuildSharedLinearLayout()` + `BuildSharedTensor()` | Construct shared-memory clang AST types from `SharedTensorParameter` | Med | `clang_compiler.cc` (new methods on TypeBuilder), `clang_compiler.h:137` (new data struct) |
| `TypeInspector` shared-tensor parsing (`ParseSharedTensorType`) | Reverse clang AST → `SharedTensorParameter` for return-type inference compatibility | Med | `clang_compiler.h:263` (new methods on TypeInspector) |
| `extractExternCallSpecs()` handles `MemDescType` | Extracts shared-memory layout info from MLIR ops for CUDA compilation | Med | `clang_compiler.cc:1152-1219` (type dispatch change), `compiler.py:786` (JSON parsing) |
| `SharedMemoryObject::getBase()` passed as callee arg in lowering | The shared-memory `ptr addrspace(3)` reaches the device function | Low | `ExternCallOpToLLVM.cpp:141-152` (new arg path, parallel to alloca+store) |
| Callee LLVM signature matches `ptr addrspace(3)` for shared args | Clang naturally emits addrspace(3) for `__device__` shared params — must verify compatibility | Low | `clang_compiler.cc` (no code change needed; verification-only) |
| Frontend `call_extern()` allows `shared_memory_descriptor` args | User-facing API must pass shared memory descriptors alongside tensors | Low | `_semantic.py:254` (relax isinstance check) |
| E2E test: shared-memory read+write through `gl.call()` | Validates the full pipeline | Low | `python/test/gluon/test_extern_call.py` (new test case) |

## Differentiators

Features that set this capability apart. Not required for MVP but differentiates from alternative approaches.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `SharedLinearLayout` with full arbitrary swizzle (offset_bases + block_bases + alignment) | The existing `SwizzledSharedEncoding` is limited to strided patterns with perPhase/maxPhase; `SharedLinearLayout` supports arbitrary basis matrices — any swizzle pattern expressible as a linear map | Med | Already exists as `SharedLinearEncodingAttr`; needs C++ template counterpart |
| Mutable shared-memory parameter (`SharedTensor&` vs `const SharedTensor&`) | The callee can read AND write shared memory — enables reduction buffers, scratch space, atomics through `gl.call()` | Low | `SharedMemorySpaceAttr` already sets `mutableMemory=true` at `gluon_ir.cc:315` |
| Integration with v1.0 return-type inference (transparent to shared-memory args) | `TypeInspector::DispatchTypeParsing()` already handles mixed `Tensor`/`Tuple` return types; adding `SharedTensor` recognition does not change the inference flow | Low | `FunctionResolver` template deduction will handle `SharedTensor` args the same as `Tensor` args |
| Reuse of existing `ModuleAllocation` + `AllocateSharedMemory` pass | No new allocation analysis needed — shared memory for `gl.call()` arguments uses the same allocation offsets as all other shared-memory ops | Med | `AllocateSharedMemory.cpp:17-26` already handles all `MemDescType` ops |

## Anti-Features

What NOT to build in this milestone.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Returning a `shared_memory_descriptor` from `gl.call()` | Much larger scope: requires tracking shared-memory liveness across the caller/callee boundary; introduces allocation-lifetime concerns | Deferred to future milestone per PROJECT.md:63 decision |
| A new MLIR dialect op for extern-call-with-shared-memory | Fragments the `gl.call()` code path; creates duplicate lowering, spec extraction, and Python binding code | Extend `ttg.extern_call` to accept mixed tensor/memdesc operands |
| Wrapper LLVM functions to type-convert shared-memory pointers | Adds indirection; the types already match (both sides are `ptr addrspace 3`) | Pass `SharedMemoryObject::getBase()` directly; clang's natural CodeGen produces matching signatures |
| `result_layout` derivation from shared-memory layout | Shared-memory args don't produce return types; this is a separate inference concern | Return-type inference stays unchanged for v1.1 |
| Dynamic shared memory allocation (variable-size smem) from `gl.call()` | Would require CUDA dynamic-shared-memory ABI support (`extern __shared__`); not needed for fixed-shape shared memory | Use only fixed-shape `SharedLinearLayout` based on existing `local_alloc` allocations |

## Feature Dependencies

```
SharedLinearLayout C++ template
    → SharedTensor<T,Shape,SharedLinearLayout> C++ template
        → TypeBuilder::BuildSharedTensor() (clang AST construction)
            → CudaFuncRequest extended with SharedTensorParameter
                → _pre_compile_extern_calls() builds shared-mem CUDA requests
                    → CUDA bitcode with SharedTensor& callee signature

extern_call ODS relaxed (TT_Tensor → AnyTypeOf<[TT_Tensor, TTG_MemDescType]>)
    → extractExternCallSpecs() handles MemDescType (type dispatch)
        → SpecInput extended with "memory_space": "shared" + offset_bases/block_bases/alignment
            → _pre_compile_extern_calls() builds SharedTensorParameter from shared specs
                → [converges with CUDA path above]

ExternCallOpToLLVM shared-memory lowering
    → [depends on] extern_call ODS relaxed (must have memdesc operands to lower)
    → [depends on] CUDA bitcode with SharedTensor& callee (must have matching function to call)
    → getSharedMemoryObjectFromStruct() → getBase() → passed as callee arg

Frontend call_extern() arg validation
    → [depends on] all of the above (end-to-end integration)
```

## MVP Recommendation

Prioritize:
1. **C++ templates** (`SharedLinearLayout`, `SharedTensor`) in `tt_plugin.cu` — the C++ foundation
2. **`extern_call` ODS relaxation** — unblocks all downstream MLIR work
3. **`extractExternCallSpecs()` + `_pre_compile_extern_calls()` extension** — connects MLIR to CUDA compilation
4. **TypeBuilder/TypeInspector shared-tensor paths** — enables CUDA compilation from MLIR specs
5. **ExternCallOpToLLVM shared-memory arg lowering** — the LLVM-level integration

Defer:
- **Frontend syntactic sugar** (auto-detecting shared-memory args in `call_extern()`): simple validation change, do last
- **E2E tests**: run after the pipeline is assembled

## Sources

- `TritonGPUOps.td:786-814` — ExternCallOp ODS definition (`Variadic<TT_Tensor>:$inputs`)
- `TritonGPUAttrDefs.td:395-426` — SharedLinearEncodingAttr ODS
- `gluon_ir.cc:308-318` — `get_shared_mem_desc_ty` (MemDescType builder)
- `gluon_ir.cc:454-466` — `get_shared_linear_layout` (SharedLinearEncodingAttr builder)
- `_layouts.py:631-673` — SharedLinearLayout Python dataclass
- `tt_plugin.cu:1-99` — Existing C++ Shape, TensorLayout, Tensor templates
- `clang_compiler.h:129-141,224-280` — TensorParameter, TypeBuilder, TypeInspector
- `clang_compiler.cc:1152-1219` — extractExternCallSpecs (MLIR scan)
- `compiler.py:709-807` — _pre_compile_extern_calls (CUDA compilation orchestration)
- `ExternCallOpToLLVM.cpp:111-293` — Extern call LLVM lowering
- `Utility.cpp:1569-1589` — getSharedMemoryBase (addrspace-3 ptr construction)
- `Utility.h:374-448` — SharedMemoryObject class
