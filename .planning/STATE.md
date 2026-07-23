---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Shared Memory Interop
current_phase: 07
current_phase_name: e2e-verification
status: complete
stopped_at: Phase 07 verified — 19/19 pass, 0 fail
last_updated: "2026-07-23T04:50:00.000Z"
last_activity: 2026-07-23
last_activity_desc: Phase 07 E2E verification complete (lit + GPU tests)
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 10
  completed_plans: 10
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-15)

**Core value:** `gl.call()` produces MLIR result types (dtype, shape, layout) matching what the CUDA C++ function actually returns, with type-consistent downstream IR — extended so Gluon `shared_memory_descriptor` buffers can be passed into device functions as `SharedTensor<T,Shape,SharedLinearLayout>&` with correct addrspace-3 lowering.
**Current focus:** Phase 07 — e2e-verification

## Current Position

Phase: 07 (e2e-verification) — VERIFIED ✓
Plan: 2 of 2 — COMPLETE
Status: All tests pass — 19 pass, 0 fail (12 GPU E2E + 1 pybind + 6 lit)
Last activity: 2026-07-23 — Phase 07 E2E verification complete

### Phase 07 Deliverables

- `tt_plugin.cu`: `shared_accumulate`, `write_swizzled_2d` device functions
- `test_extern_call.py`: 6 new test functions + 3 kernel functions (167→340+ lines)
- `gl.call()` scalar constexpr arg support (op attributes pipeline)
- Path portability fixes (LLVM_SYSPATH, CUDA_HOME env vars)

Progress: [██████████] 100% (v1.1 phases)

## Performance Metrics

**Velocity:**

- Total plans completed: 22 (across v1.0)
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Seam & Cleanup | 2 | — | — |
| 2. Semantic-Time Inference | 5 | — | — |
| 3. Verification | 1 | — | — |
| Phase 04-c-templates-clang-ast-foundation P01 | 8min | 3 tasks | 4 files |
| Phase 04-c-templates-clang-ast-foundation P02 | 7min | 3 tasks | 2 files |
| Phase 04-c-templates-clang-ast-foundation P03 | 15min | 3 tasks | 1 files |
| 04 | 3 | - | - |
| Phase 05-mlir-op-relaxation-spec-extraction P01 | 9min | 3 tasks | 3 files |
| Phase 05-mlir-op-relaxation-spec-extraction P02 | 6 | 4 tasks | 1 files |
| 05 | 2 | - | - |
| Phase 06-cuda-wiring-llvm-lowering-frontend-api P01 | 1 min | 2 tasks | 2 files |
| Phase 06-cuda-wiring-llvm-lowering-frontend-api P02 | 4min | 4 tasks | 2 files |
| Phase 06-cuda-wiring-llvm-lowering-frontend-api P03 | 4 min | 3 tasks | 2 files |
| 06 | 3 | - | - |
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 07 P01 | 12 min | 1 tasks | 2 files |
| Phase 07-e2e-verification P02 | 43 min | 4 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent:

- Keep `result_layout` required (explicit final layout), not auto-derived.
- Infer shape/dtype/layout at semantic (IR-build) time for type-consistent downstream IR.
- Reach CUDA inference from the Gluon frontend via the backend `codegen_fns` hook.
- SharedTensor is argument-only for v1.1 (returning shared memory deferred to SHRET-01).
- New C++ `SharedLinearLayout` distinct from distributed `Layout` (different addressing model) — validated Phase 4, D-07 swizzle parity proven.
- `SharedTensor<T,Shape,SharedLinearLayout>` is a separate C++ template, not `Tensor` with a different layout parameter.
- OffsetBases/BlockBases use RANK+N_BASES NTTP carrier structs (C++20 structural type requirement).
- Swizzle parity verified via static_assert in synthetic .cu (parse-only verification avoids pre-existing coroutine crash).
- [Phase ?]: Used std::variant<TensorSpecInput, SharedSpecInput> instead of optional fields on a single struct — per D-10, cleaner type-level separation
- [Phase ?]: GPU E2E regression (Task 3/SHTEST-03) deferred due to pre-existing LLVM dynamic-linking build issue

### Pending Todos

None yet.

### Blockers/Concerns

- **Phase 5 (ODS relaxation):** `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>` has the widest blast radius — any downstream pass that assumes tensor-only inputs must be verified.
- **Phase 6 (lowering):** Address-space mismatch (alloca/store produces addrspace 0; callee expects addrspace 3) causes silent data corruption — LLVM IR dump inspection is mandatory for verification.
- **[Phase 4] Pre-existing CUDACompiler coroutine destructor segfault:** `infer()`/`compileBitcode()` crash when the compiler is destroyed before its LLVMContext outside the gluon.jit pipeline; tests work around via module-level compiler/context caching (mirrors `InferExternCallResult._compilers`). Root cause in X64SysVABI coroutine stack ownership — watch in Phases 6-7.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Auto layout | Make `result_layout` optional/auto-derived (AUTO-01) | Deferred | 2026-07-11 |
| Precision | Full Fp64 pipeline support (FP64-01) | Deferred | 2026-07-11 |
| Shared return | Return `shared_memory_descriptor` from `gl.call()` (SHRET-01) | Deferred | 2026-07-12 |
| Padded layout | `PaddedSharedLayout` shared encoding support | Deferred | 2026-07-12 |

## Session Continuity

Last session: 2026-07-21T16:35:53.428Z
Stopped at: Completed 07-02-PLAN.md
Resume file: None
