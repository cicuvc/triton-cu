# Roadmap: triton-cu CUDA C++ Interop for gl.call()

## Milestones

- ✅ **v1.0 Return Type Inference** — Phases 1-3 (shipped 2026-07-12) — see [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)
- 🚧 **v1.1 Shared Memory Interop** — Phases 4-7 (planning)

## Phases

<details>
<summary>✅ v1.0 Return Type Inference (Phases 1-3) — SHIPPED 2026-07-12</summary>

### Phase 1: Seam & Cleanup

**Goal**: Build the inference seam (backend `codegen_fns` hook + suspended `CUDACompiler`) and fix known bugs
**Depends on**: Nothing (first phase)
**Requirements**: infer-seam, f64-guard
**Success Criteria**:

  1. `InferExternCallResult` object is exposed via CUDA backend `codegen_fns` hook
  2. Suspended `CUDACompiler` parses `.cu` once at semantic time and resumes at llir stage
  3. Frontend and backend both raise `NotImplementedError` for f64/fp64/float64 (no silent coercion)
  4. Dead unreachable code at `compiler.py:510-513` is removed
  5. All 4 existing extern-call tests pass unchanged

**Plans**: 2 plans

Plans:

- [x] 01-01: Build the inference seam (hook + suspended compiler)
- [x] 01-02: Fix known bugs (dead code, f64 guard)

### Phase 2: Semantic-Time Inference

**Goal**: CUDA-inferred return dtype, shape, and layout flow into the `ttg.extern_call` op result type at IR-build time
**Depends on**: Phase 1
**Requirements**: INFER-01, INFER-02, INFER-03, INFER-04, INFER-05
**Success Criteria**:

  1. `call_extern` consumes CUDA-inferred dtype+shape via `infer_hook.infer_result()` with `first_input` fallback for fixed-layout functions
  2. Fixed-layout `reduce` obtains return dtype+shape from CUDA inference via `PlaceholderLayout` + `ExplicitTemplateArgs` fallback
  3. `call_extern` raises a clear `RuntimeError` when the `infer_extern_call_result` hook is absent (non-CUDA backend), with an automated test verifying the raise
  4. `result_layout=` remains the requested final layout; `convert_layout` reconciles CUDA-native → user layout
  5. `ttgir` dump confirms correct dtype + shape in the op result type for a shape-and-dtype-changing extern call

**Plans**: 5 plans

Plans:

- [x] 02-01: PlaceholderLayout + BuildTensor placeholder mode
- [x] 02-02: Pybind11 infer() binding + InferExternCallResult.infer_result()
- [x] 02-03: call_extern consumes CUDA-inferred dtype+shape
- [x] 02-04: Fixed-layout resolve via PlaceholderLayout + ExplicitTemplateArgs
- [x] 02-05: Bundle remaining cleanup (dead code, hook-absent raise, test)

### Phase 3: Verification

**Goal**: E2E correctness proven — a shape-and-dtype-changing extern call works, no regressions
**Depends on**: Phase 2
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria**:

  1. New E2E test (`test_reduce_f16_f32`) passes — `gl.call()` handles simultaneous shape AND dtype transitions (f16→f32 reduce)
  2. All 4 existing extern-call tests pass unchanged (6/6 total including new + hook test)
  3. Gluon lit suite (5/5 tests) passes unchanged — zero MLIR/dialect/production source changes

**Plans**: 1 plan

Plans:

- [x] 03-01: E2E test + regression verification

</details>

### 🚧 v1.1 Shared Memory Interop (Planning)

**Milestone Goal:** Gluon `shared_memory_descriptor` buffers can be passed into CUDA C++ `__device__` template functions via `gl.call()` as `SharedTensor<T,Shape,SharedLinearLayout>&` with correct addrspace-3 lowering, read+write access, full swizzle support, and integration with the v1.0 return-type inference machinery.

- [ ] **Phase 4: C++ Templates + Clang AST Foundation** — Device-side SharedLinearLayout/SharedTensor templates and clang AST round-trip infrastructure
- [ ] **Phase 5: MLIR Op Relaxation + Spec Extraction** — Relax ttg.extern_call ODS for MemDescType; extract shared-layout JSON specs
- [ ] **Phase 6: CUDA Wiring + LLVM Lowering + Frontend API** — Wire shared args through compilation; per-operand ptr addrspace(3) lowering; frontend accepts shared_memory_descriptor
- [ ] **Phase 7: E2E Verification** — Shared read+write GPU test, swizzle-correctness test, full regression

## Phase Details

### Phase 4: C++ Templates + Clang AST Foundation

**Goal**: Shared memory device types and clang AST bridge exist — a standalone `.cu` file compiles with `SharedTensor&` parameters, and the clang infrastructure round-trips `SharedTensorParameter` through AST construction → inspection
**Depends on**: Nothing (first v1.1 phase)
**Milestone**: v1.1
**Requirements**: SHTYPE-01, SHTYPE-02, SHAST-01, SHAST-02, SHAST-03
**Success Criteria** (what must be TRUE):

  1. A standalone `.cu` device function taking `SharedTensor<float, Shape<2>, SharedLinearLayout<OffsetBases{...}, BlockBases{...}, 16>>&` compiles successfully with nvcc/clang — the template is syntactically and semantically valid
  2. `TypeBuilder::BuildSharedTensor()` constructs a valid clang AST node from a `SharedTensorParameter` — verified via `-ast-dump` showing the correct template specialization with scalar type, shape dims, and layout bases
  3. `TypeInspector::DispatchTypeParsing()` parses a `SharedTensor<...>&` AST node back to a `SharedTensorParameter` with scalar type, shape dims, and layout bases matching the original input — round-trip verification passes
  4. `FunctionResolver::LookupFunction()` resolves a `__device__` template function with `SharedTensor&` parameters via clang Sema template argument deduction — no substitution failure or ambiguity error
  5. Python `llvm.SharedTensorParameter` pybind11 class is importable and constructable with `.type`, `.shape`, `.offset_bases`, `.block_bases`, `.alignment` attributes

**Plans**: 3 plans

Plans:
**Wave 1**

- [x] 04-01-PLAN.md — Device templates (SharedLinearLayout, SharedTensor in tt_plugin.cu) + C++ structs (SharedLayoutInfo, SharedTensorParameter in clang_compiler.h) + pybind binding (llvm.SharedTensorParameter in llvm.cc)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 04-02-PLAN.md — Clang AST implementations: TypeBuilder::BuildSharedLinearLayout/BuildSharedTensor, TypeInspector::ParseSharedTensorType/DispatchTypeParsing, CUDACompiler::BuildSharedTensor + variant dispatch

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 04-03-PLAN.md — GPU-free pytest harness: round-trip verification, FunctionResolver test, D-07 swizzle-parity test

**UI hint**: no

### Phase 5: MLIR Op Relaxation + Spec Extraction

**Goal**: The `ttg.extern_call` op accepts mixed tensor+memdesc operands, and `extractExternCallSpecs()` emits shared-layout JSON for memdesc operands without crashing
**Depends on**: Phase 4
**Milestone**: v1.1
**Requirements**: SHMLIR-01, SHMLIR-02
**Success Criteria** (what must be TRUE):

  1. A lit test with `ttg.extern_call` taking mixed `TT_Tensor` and `TTG_MemDescType` inputs passes MLIR verification — no `Variadic` type constraint error and no downstream verifier failure
  2. `extractExternCallSpecs()` processes a `MemDescType` operand without the `cast<RankedTensorType>` crash — the `dyn_cast` branch extracts shared layout correctly
  3. The emitted JSON spec for a memdesc operand contains `"memory_space": "shared"`, `"offset_bases"`, `"block_bases"`, and `"alignment"` keys with values matching the MLIR `SharedLinearEncodingAttr`
  4. Tensor-only `extern_call` specs continue to work unchanged — the existing distributed-layout extraction path is unaffected by the new branch

**Plans**: TBD
**UI hint**: no

### Phase 6: CUDA Wiring + LLVM Lowering + Frontend API

**Goal**: Shared-memory args flow end-to-end from `gl.call()` through CUDA compilation to LLVM lowering with correct `ptr addrspace(3)` emission, and the frontend accepts `shared_memory_descriptor` arguments
**Depends on**: Phase 4, Phase 5
**Milestone**: v1.1
**Requirements**: SHWIRE-01, SHLOWER-01, SHLOWER-02, SHAPI-01
**Success Criteria** (what must be TRUE):

  1. `_pre_compile_extern_calls()` builds `SharedTensorParameter` for shared inputs and routes them through the suspended-compiler path — the single-parse guard assertion at `compiler.py:683` holds (parse count unchanged)
  2. `gl.call()` in `_semantic.py` accepts a `shared_memory_descriptor` alongside tensor arguments — no `TypeError` or validation rejection
  3. Emitted LLVM IR shows `ptr addrspace(3)` for shared-memory argument positions in the `call` instruction, and `ptr addrspace(0)` for tensor positions — verified via LLVM IR dump inspection of a mixed-arg call site
  4. When a shared memory descriptor with accumulated subview offsets (from `.slice()` / `.index()`) is passed, the callee receives `base + shmemOffset` (GEP-computed subview address), not the raw allocation base — verified via LLVM IR showing the correct GEP offset

**Plans**: TBD
**UI hint**: no

### Phase 7: E2E Verification

**Goal**: Full pipeline works end-to-end — shared memory read+write through `gl.call()` produces correct GPU results, swizzle layouts round-trip correctly, and all existing tests pass without regression
**Depends on**: Phase 6
**Milestone**: v1.1
**Requirements**: SHTEST-01, SHTEST-02, SHTEST-03
**Success Criteria** (what must be TRUE):

  1. New E2E GPU test passes: a Gluon kernel allocates shared memory, passes the descriptor to a CUDA device function that reads AND writes it (with explicit `gl.barrier()` synchronization before and after), and GPU output matches the reference within tolerance on the RTX 5090
  2. Swizzle-correctness test passes: values written through `SharedTensor&` at specific logical indices using a non-trivial swizzled `SharedLinearLayout` are read back via `shared_memory_descriptor.load()` bit-for-bit correctly, with each basis (offset/block) exercised independently
  3. All 6 existing `test_extern_call.py` tests pass unchanged — no regression in the tensor-only path
  4. The Gluon lit suite (5/5 tests) passes unchanged — no MLIR/dialect-level regression

**Plans**: TBD
**UI hint**: no

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Seam & Cleanup | v1.0 | 2/2 | Complete | 2026-07-11 |
| 2. Semantic-Time Inference | v1.0 | 5/5 | Complete | 2026-07-11 |
| 3. Verification | v1.0 | 1/1 | Complete | 2026-07-11 |
| 4. C++ Templates + Clang AST Foundation | v1.1 | 0/3 | Not started | - |
| 5. MLIR Op Relaxation + Spec Extraction | v1.1 | 0/TBD | Not started | - |
| 6. CUDA Wiring + LLVM Lowering + Frontend API | v1.1 | 0/TBD | Not started | - |
| 7. E2E Verification | v1.1 | 0/TBD | Not started | - |
