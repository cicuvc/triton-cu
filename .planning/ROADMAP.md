# Roadmap: triton-cu Return Type Inference for gl.call()

## Overview

Complete the return-type inference feature for `gl.call()`. The CUDA-side inference machinery already exists and is wired up to the point where it patches the op's *layout*; the remaining work is to make the CUDA-inferred **shape and dtype** flow into the op's result type at IR-build time so downstream MLIR stays type-consistent. We first build the frontend↔backend seam and clean up bundled bugs (Phase 1), then plumb inferred shape/dtype/layout through the semantic layer with layout reconciliation (Phase 2), then verify end-to-end with new and existing tests (Phase 3).

## Phases

- [x] **Phase 1: Seam & Cleanup** - Backend `codegen_fns` inference hook + single-parse plumbing + bundled bug fixes (completed 2026-07-11)
- [ ] **Phase 2: Semantic-Time Inference** - Shape/dtype/layout inference at IR-build time with layout reconciliation
- [ ] **Phase 3: Verification** - New shape/dtype-changing test + regression + module verification

## Phase Details

### Phase 1: Seam & Cleanup

**Goal**: Establish a clean way for the backend-agnostic Gluon semantic layer to invoke CUDA return-type inference, ensure the `.cu` is not parsed twice, and clear the bundled bugs — before touching the inference data flow.
**Depends on**: Nothing (first phase)
**Requirements**: INFER-06, INFER-07, BUG-01, BUG-02
**Success Criteria** (what must be TRUE):

  1. The CUDA backend exposes a return-type-inference callable via `get_codegen_implementation` / `codegen_fns` that the Gluon semantic layer can call, given `(src_path, func, arg tensor types/layouts, use_fast_math)`.
  2. Non-CUDA/interpreter paths degrade gracefully (no crash when the hook is absent).
  3. Inference and the existing `llir`-stage bitcode compilation share a single clang parse of a given `.cu` (reuse/cache), or the single-parse path is documented and measured.
  4. Dead code at `compiler.py:510-513` is removed; `f64`/`fp64` handling is either a clear error or explicitly documented.
  5. Existing 4 extern-call tests still pass (no behavior change yet).

**Plans**: 2/2 plans complete

Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Bug fixes: remove dead code (BUG-01) and add f64/fp64 guard at both layers (BUG-02)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Inference hook & single-parse seam: InferExternCallResult via codegen_fns (INFER-06), suspended CUDACompiler + parse-counter assertion (INFER-07)

### Phase 2: Semantic-Time Inference

**Goal**: Make `call_extern` (in `_semantic.py`) obtain the CUDA-inferred element type, shape, and native layout, build the `ttg.extern_call` result type from them, and reconcile to the user's requested `result_layout` via `convert_layout`.
**Depends on**: Phase 1
**Requirements**: INFER-01, INFER-02, INFER-03, INFER-04, INFER-05
**Success Criteria** (what must be TRUE):

  1. `call_extern` builds each result's `distributed_type` from CUDA-inferred dtype + shape + native layout (via `TensorParameter` → `DistributedLinearLayout` round-trip), not from `first_input`.
  2. When the user's `result_layout` differs from the CUDA-native layout, a `convert_layout` produces the final tensor in the requested layout; downstream ops consume the user layout.
  3. `assert_no_conv=True` raises when a conversion would be required.
  4. A kernel calling a shape-changing extern function (e.g. `reduce`) compiles and lowers with `llvm.verify_module` passing, WITHOUT the user hand-matching the return shape.
  5. Multi-return (`std::tuple`) and existing same-shape cases continue to work.

**Plans**: 1/3 plans executed

Plans:
**Wave 1**

- [x] 02-01-PLAN.md — Device library + C++ core: PlaceholderLayout in tt_plugin.cu, BuildTensor placeholder mode, CUDACompiler::inferReturnTypes (INFER-01, INFER-02)

**Wave 2** *(blocked on Wave 1)*

- [ ] 02-02-PLAN.md — Python bindings + hook: SuspendedCudaCompiler.infer() binding, InferExternCallResult.infer_result() (INFER-03)

**Wave 3** *(blocked on Wave 2)*

- [ ] 02-03-PLAN.md — Semantic-time consumption: call_extern hook integration, build + regression verification (INFER-04, INFER-05)

### Phase 3: Verification

**Goal**: Prove the feature end-to-end and guard against regressions.
**Depends on**: Phase 2
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):

  1. A new `test_extern_call.py` test exercises an extern call whose return shape and/or dtype differs from the first argument, supplying only the final `result_layout` (not a hand-computed shape/dtype), and produces numerically correct results on GPU.
  2. All 4 existing extern-call tests pass unchanged.
  3. `llvm.verify_module` passes after extern linking for new and existing cases; lit suite is unaffected.

**Plans**: TBD

Plans:

- [ ] 03-01: TBD during planning

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Seam & Cleanup | 2/2 | Complete    | 2026-07-11 |
| 2. Semantic-Time Inference | 1/3 | In Progress|  |
| 3. Verification | 0/TBD | Not started | - |
