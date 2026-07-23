# Roadmap: triton-cu CUDA C++ Interop for gl.call()

## Milestones

- ✅ **v1.0 Return Type Inference** — Phases 1-3 (shipped 2026-07-12) — see [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)
- ✅ **v1.1 Shared Memory Interop** — Phases 4-7 (shipped 2026-07-23) — see [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)

## Phases

<details>
<summary>✅ v1.0 Return Type Inference (Phases 1-3) — SHIPPED 2026-07-12</summary>

### Phase 1: Seam & Cleanup

**Goal**: Build the inference seam (backend `codegen_fns` hook + suspended `CUDACompiler`) and fix known bugs
**Depends on**: Nothing (first phase)
**Requirements**: infer-seam, f64-guard
**Plans**: 2/2 plans complete

### Phase 2: Semantic-Time Inference

**Goal**: CUDA-inferred return dtype, shape, and layout flow into the `ttg.extern_call` op result type at IR-build time
**Depends on**: Phase 1
**Requirements**: INFER-01, INFER-02, INFER-03, INFER-04, INFER-05
**Plans**: 5/5 plans complete

### Phase 3: Verification

**Goal**: E2E correctness proven — a shape-and-dtype-changing extern call works, no regressions
**Depends on**: Phase 2
**Requirements**: TEST-01, TEST-02, TEST-03
**Plans**: 1/1 plan complete

</details>

<details>
<summary>✅ v1.1 Shared Memory Interop (Phases 4-7) — SHIPPED 2026-07-23</summary>

### Phase 4: C++ Templates + Clang AST Foundation

**Goal**: Shared memory device types and clang AST bridge exist — a standalone `.cu` file compiles with `SharedTensor&` parameters, and the clang infrastructure round-trips `SharedTensorParameter` through AST construction → inspection
**Depends on**: Nothing (first v1.1 phase)
**Requirements**: SHTYPE-01, SHTYPE-02, SHAST-01, SHAST-02, SHAST-03
**Plans**: 3/3 plans complete

### Phase 5: MLIR Op Relaxation + Spec Extraction

**Goal**: The `ttg.extern_call` op accepts mixed tensor+memdesc operands, and `extractExternCallSpecs()` emits shared-layout JSON for memdesc operands without crashing
**Depends on**: Phase 4
**Requirements**: SHMLIR-01, SHMLIR-02
**Plans**: 2/2 plans complete

### Phase 6: CUDA Wiring + LLVM Lowering + Frontend API

**Goal**: Shared-memory args flow end-to-end from `gl.call()` through CUDA compilation to LLVM lowering with correct `ptr addrspace(3)` emission, and the frontend accepts `shared_memory_descriptor` arguments
**Depends on**: Phase 4, Phase 5
**Requirements**: SHWIRE-01, SHLOWER-01, SHLOWER-02, SHAPI-01
**Plans**: 3/3 plans complete

### Phase 7: E2E Verification

**Goal**: Full pipeline works end-to-end — shared memory read+write through `gl.call()` produces correct GPU results, swizzle layouts round-trip correctly, and all existing tests pass without regression
**Depends on**: Phase 6
**Requirements**: SHTEST-01, SHTEST-02, SHTEST-03
**Plans**: 2/2 plans executed, verified ✓ — 19/19 pass, 0 fail

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Seam & Cleanup | v1.0 | 2/2 | Complete | 2026-07-11 |
| 2. Semantic-Time Inference | v1.0 | 5/5 | Complete | 2026-07-11 |
| 3. Verification | v1.0 | 1/1 | Complete | 2026-07-11 |
| 4. C++ Templates + Clang AST Foundation | v1.1 | 3/3 | Complete | 2026-07-12 |
| 5. MLIR Op Relaxation + Spec Extraction | v1.1 | 2/2 | Complete | 2026-07-15 |
| 6. CUDA Wiring + LLVM Lowering + Frontend API | v1.1 | 3/3 | Complete | 2026-07-15 |
| 7. E2E Verification | v1.1 | 2/2 | Complete ✓ | 2026-07-23 |
