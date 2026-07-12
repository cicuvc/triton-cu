---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Shared Memory Interop
current_phase: 4
current_phase_name: C++ Templates + Clang AST Foundation
status: executing
stopped_at: Phase 4 context gathered
last_updated: "2026-07-12T13:16:25.358Z"
last_activity: 2026-07-12
last_activity_desc: Roadmap created for v1.1 Shared Memory Interop (4 phases)
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-12)

**Core value:** `gl.call()` produces MLIR result types (dtype, shape, layout) matching what the CUDA C++ function actually returns, with type-consistent downstream IR — extended so Gluon `shared_memory_descriptor` buffers can be passed into device functions as `SharedTensor<T,Shape,SharedLinearLayout>&` with correct addrspace-3 lowering.
**Current focus:** Phase 4 — C++ Templates + Clang AST Foundation (v1.1 milestone)

## Current Position

Phase: 4 of 7 (C++ Templates + Clang AST Foundation)
Plan: —
Status: Ready to execute
Last activity: 2026-07-12 — Roadmap created for v1.1 Shared Memory Interop (4 phases)

Progress: [████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░] 0% (v1.1 phases)

## Performance Metrics

**Velocity:**

- Total plans completed: 8 (across v1.0)
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Seam & Cleanup | 2 | — | — |
| 2. Semantic-Time Inference | 5 | — | — |
| 3. Verification | 1 | — | — |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent:

- Keep `result_layout` required (explicit final layout), not auto-derived.
- Infer shape/dtype/layout at semantic (IR-build) time for type-consistent downstream IR.
- Reach CUDA inference from the Gluon frontend via the backend `codegen_fns` hook.
- SharedTensor is argument-only for v1.1 (returning shared memory deferred to SHRET-01).
- New C++ `SharedLinearLayout` distinct from distributed `Layout` (different addressing model).
- `SharedTensor<T,Shape,SharedLinearLayout>` is a separate C++ template, not `Tensor` with a different layout parameter.

### Pending Todos

None yet.

### Blockers/Concerns

- **Phase 5 (ODS relaxation):** `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>` has the widest blast radius — any downstream pass that assumes tensor-only inputs must be verified.
- **Phase 6 (lowering):** Address-space mismatch (alloca/store produces addrspace 0; callee expects addrspace 3) causes silent data corruption — LLVM IR dump inspection is mandatory for verification.
- **Swizzle parity (Phase 4 + 7):** The C++ `SharedLinearLayout::evaluate()` byte-offset formula must be bit-identical to the MLIR `LinearLayout({offsetBases, blockBases}, outDims)` composition at `gluon_ir.cc:102-103` — mismatch means wrong shared-memory addresses.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Auto layout | Make `result_layout` optional/auto-derived (AUTO-01) | Deferred | 2026-07-11 |
| Precision | Full Fp64 pipeline support (FP64-01) | Deferred | 2026-07-11 |
| Shared return | Return `shared_memory_descriptor` from `gl.call()` (SHRET-01) | Deferred | 2026-07-12 |
| Padded layout | `PaddedSharedLayout` shared encoding support | Deferred | 2026-07-12 |

## Session Continuity

Last session: 2026-07-12T12:52:43.596Z
Stopped at: Phase 4 context gathered
Resume file: .planning/phases/04-c-templates-clang-ast-foundation/04-CONTEXT.md
