# Requirements: triton-cu — Shared Memory Interop for gl.call()

**Defined:** 2026-07-12
**Milestone:** v1.1 Shared Memory Interop
**Core Value:** `gl.call()` produces MLIR result types matching what the CUDA C++ function returns, with type-consistent downstream IR — now extended so Gluon `shared_memory_descriptor` buffers can be passed into device functions as `SharedTensor<dtype,shape,layout>&` with correct addrspace-3 lowering.

## v1.1 Requirements

Requirements for milestone v1.1. Each maps to a roadmap phase. IDs continue the project's category-based scheme (v1.0 used INFER-/TEST-).

### C++ Device Types

New CUDA C++ device-side templates in `python/test/gluon/tt_plugin.cu`, modeled on the existing `TensorLayout`/`Tensor` templates.

- [x] **SHTYPE-01**: A `SharedLinearLayout` C++ device template exists carrying `offset_bases` + `block_bases` + `alignment`, computing shared-memory byte offsets bit-identical to the MLIR shared linear-layout composition (`gluon_ir.cc:102-103`) — full arbitrary swizzle expressible as a linear basis map
- [x] **SHTYPE-02**: A `SharedTensor<dtype, shape, layout>` C++ device template exists (separate template, not `Tensor`) whose backing storage lives in shared memory (addrspace 3) and supports element **read and write**

### Clang AST Bridge

Extend the v1.0 clang type-inference infrastructure (`clang_compiler.cc`/`.h`) with a parallel shared-tensor data path.

- [x] **SHAST-01**: `SharedLayoutInfo` + `SharedTensorParameter` structs carry shared-layout data across the Python/C++ boundary; `CudaFuncRequest::ParamTypes` accepts the `SharedTensorParameter` variant
- [x] **SHAST-02**: `TypeBuilder::BuildSharedLinearLayout()` + `BuildSharedTensor()` construct the shared clang AST types from a `SharedTensorParameter` (parallel to `BuildLayout()`/`BuildTensor()`)
- [x] **SHAST-03**: `TypeInspector` parses a `SharedTensor<...>&` clang type back to a `SharedTensorParameter` (`DispatchTypeParsing` branch), and `FunctionResolver` resolves device functions with `SharedTensor&` parameters via Sema template deduction — integrating shared args into the existing return-type inference flow

### MLIR Op & Spec Extraction

Relax the `ttg.extern_call` op and its spec extractor to accept shared-memory (`MemDescType`) operands.

- [x] **SHMLIR-01**: `ttg.extern_call` accepts `MemDescType` operands (ODS relaxation at `TritonGPUOps.td:803`) without breaking the tensor-only path; a lit test verifies both tensor-only and mixed tensor+memdesc inputs pass verification
- [x] **SHMLIR-02**: `extractExternCallSpecs()` handles `MemDescType` operands (no `cast<RankedTensorType>` crash) and emits shared-layout specs (`memory_space=shared`, `offset_bases`, `block_bases`, `alignment`) via `toLinearLayout(memDescType)` — covering `SharedLinearEncodingAttr`, `SwizzledSharedEncoding`, and `NVMMASharedEncoding` (all convert to a shared linear layout)

### CUDA Compilation Wiring

Connect the MLIR-side specs to the CUDA compiler through the existing single-parse path.

- [ ] **SHWIRE-01**: `_pre_compile_extern_calls()` builds `SharedTensorParameter` for shared inputs (with a `llvm.SharedTensorParameter` pybind11 binding) and routes them through the existing suspended-compiler path, preserving the single-parse guard (`compiler.py:683` assertion holds — no double parse)

### LLVM Lowering

Lower shared-memory operands to `ptr addrspace(3)` in `ExternCallOpToLLVM.cpp`.

- [ ] **SHLOWER-01**: Shared-memory operands lower to a `ptr addrspace(3)` passed directly to the callee (via `getSharedMemoryObjectFromStruct` base), bypassing the distributed `alloca+store+ptr` path; distributed operands keep their existing path in mixed argument lists, matching the callee signature order
- [ ] **SHLOWER-02**: Accumulated subview offsets (`memdesc_subslice`/`memdesc_index`/`memdesc_reshape`) are applied via `getShmemOffset()` so the callee receives the correct sub-buffer address, not the allocation base

### Frontend API

Allow `gl.call()` to accept shared-memory descriptors.

- [ ] **SHAPI-01**: `gl.call()` accepts a `shared_memory_descriptor` argument alongside tensors (`_semantic.py:254` isinstance relaxation) and threads its shared-layout info to the inference hook via `arg_params`

### Verification

- [ ] **SHTEST-01**: New E2E GPU test in `test_extern_call.py` — a Gluon kernel allocates shared memory, passes the descriptor to a CUDA device fn that reads **and writes** it (with explicit `gl.barrier()` synchronization), and GPU output is verified on the RTX 5090
- [ ] **SHTEST-02**: Swizzle-correctness test — a non-trivial swizzled shared layout round-trips: values written through `SharedTensor&` at specific logical indices read back bit-for-bit correctly via `shared_memory_descriptor.load()`, exercising offset/block bases independently
- [ ] **SHTEST-03**: All 6 existing extern-call tests pass unchanged (no regression) and the Gluon lit suite is unaffected

## v2 / Future Requirements

Deferred beyond v1.1. Tracked but not in the current roadmap.

### Shared Memory Return

- **SHRET-01**: `gl.call()` can return a `shared_memory_descriptor` result (shared-memory return type), requiring shared-memory liveness tracking across the caller/callee boundary

### Prior Deferrals (from v1.0)

- **AUTO-01**: Make `result_layout=` optional / auto-derived from the CUDA-inferred layout
- **FP64-01**: Full `Fp64` support through the entire pipeline

## Out of Scope

Explicitly excluded from v1.1. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Returning `shared_memory_descriptor` from `gl.call()` | Larger scope: requires shared-memory liveness tracking across caller/callee boundary (decision 2026-07-12) — deferred to SHRET-01 |
| `PaddedSharedLayout` shared encoding | Padding does not map cleanly to a shared linear layout; swizzled/NVMMA do convert and are in scope, padded is not (decision 2026-07-12) |
| Dynamic / `extern __shared__` variable-size shared allocation | Requires CUDA dynamic-shared-memory ABI support; v1.1 uses fixed-shape allocations only |
| TMA / async-copy / mbarrier interop through `gl.call()` | Separate feature surface; not needed for basic shared read/write |
| Auto-inserted synchronization around shared-memory `gl.call()` | User must place `gl.barrier()`; automatic barrier insertion is a separate concern (documented gotcha) |
| A new dialect op for extern-call-with-shared-memory | Fragments the `gl.call()` path; instead relax the existing `ttg.extern_call` op |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SHTYPE-01 | Phase 4 | Complete |
| SHTYPE-02 | Phase 4 | Complete |
| SHAST-01 | Phase 4 | Complete |
| SHAST-02 | Phase 4 | Complete |
| SHAST-03 | Phase 4 | Complete |
| SHMLIR-01 | Phase 5 | Complete |
| SHMLIR-02 | Phase 5 | Complete |
| SHWIRE-01 | Phase 6 | Pending |
| SHLOWER-01 | Phase 6 | Pending |
| SHLOWER-02 | Phase 6 | Pending |
| SHAPI-01 | Phase 6 | Pending |
| SHTEST-01 | Phase 7 | Pending |
| SHTEST-02 | Phase 7 | Pending |
| SHTEST-03 | Phase 7 | Pending |

**Coverage:**

- v1.1 requirements: 14 total
- Mapped to phases: 14 ✓
- Unmapped: 0

---
*Requirements defined: 2026-07-12*
*Last updated: 2026-07-12 after initial definition*
