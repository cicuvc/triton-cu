---
phase: 01-seam-cleanup
plan: 01
subsystem: compiler
tags: [bugfix, dead-code, f64, fp64, guard, Triton, CUDA, gl.call]
requires: []
provides:
  - Dead code removed from make_llir return path (BUG-01)
  - f64/fp64 defense-in-depth guard at Gluon semantic layer (dtype-string, backend-agnostic)
  - _scalar_type_for helper replacing silent f64→Fp32 coercion in CUDA backend (BUG-02)
affects:
  - Phase 01-02 (inference hook uses cleaned compiler.py)
  - Phase 02 (semantic-time inference with type-safe dtype pipeline)
  - Phase 03 (verification of f64 error behavior)
tech-stack:
  added: []
  patterns:
    - "Backend-agnostic dtype-string guard: no CUDA imports in Gluon semantic layer"
    - "Explicit error over silent coercion for unsupported types"
key-files:
  created: []
  modified:
    - third_party/nvidia/backend/compiler.py
    - python/triton/experimental/gluon/language/_semantic.py
key-decisions:
  - "BUG-02 resolved as raise-NotImplementedError for f64/fp64/float64 at both layers (defense in depth)"
  - "Frontend guard is a pure dtype-string check preserving backend-agnostic layering"
patterns-established:
  - "Defense-in-depth: frontend dtype-string guard + backend _scalar_type_for backstop"
  - "Explicit error messages reference deferred FP64-01 for traceability"
requirements-completed:
  - BUG-01
  - BUG-02
coverage:
  - id: D1
    description: "Dead unreachable code at compiler.py:510-513 removed (BUG-01)"
    requirement: BUG-01
    verification:
      - kind: other
        ref: "grep -c 'del llvm_mod' third_party/nvidia/backend/compiler.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "f64/fp64 guard in call_extern raises NotImplementedError before IR build (BUG-02, Layer 1)"
    requirement: BUG-02
    verification:
      - kind: other
        ref: "grep -c 'float64' python/triton/experimental/gluon/language/_semantic.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "Backend _scalar_type_for replaces silent f64→Fp32 coercion (BUG-02, Layer 2)"
    requirement: BUG-02
    verification:
      - kind: other
        ref: "grep -c 'f64.*Fp32' third_party/nvidia/backend/compiler.py"
        status: pass
    human_judgment: false
  - id: D4
    description: "All 4 existing extern-call tests pass unchanged (regression gate)"
    requirement: BUG-02
    verification:
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py -x --tb=short"
        status: pass
    human_judgment: false
duration: 1 min
completed: 2026-07-11
status: complete
---

# Phase 01 Plan 01: Bug Fixes — Dead Code Removal & f64/fp64 Guard Summary

**Removed dead unreachable code and added defense-in-depth f64/fp64 NotImplementedError guards at both Gluon semantic and CUDA backend layers**

## Performance

- **Duration:** 1 min
- **Started:** 2026-07-11T10:31:27Z
- **Completed:** 2026-07-11T10:33:11Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Removed unreachable duplicate return block at `compiler.py:510-513` (BUG-01) — the single `return ret` at line 508 is now the only exit from `make_llir`
- Added backend-agnostic f64/fp64/float64 dtype-string guard in `call_extern` (`_semantic.py`) that raises `NotImplementedError` BEFORE building IR — no CUDA import, preserves layering
- Replaced silent f64→Fp32 coercion rows in `dtype_to_scalar` with explicit `_scalar_type_for` helper that raises `NotImplementedError` — defense in depth
- All 4 existing extern-call tests pass unchanged

## Task Commits

1. **Task 1: Remove dead unreachable code** — `1b9e4947a1` (fix)
2. **Task 2: Add f64/fp64 guard — both layers** — `bdf7d01a88` (fix)

## Files Created/Modified
- `third_party/nvidia/backend/compiler.py` — Dead code removed (lines 510-513), f64/fp64 entries removed from `dtype_to_scalar`, `_scalar_type_for` helper added, call site updated
- `python/triton/experimental/gluon/language/_semantic.py` — f64 dtype-string guard inserted before `first_input = args[0]`

## Decisions Made
- BUG-02 resolved as `raise NotImplementedError` for f64/fp64/float64 at both layers (defense in depth). Error message references deferred FP64-01 for traceability.
- Frontend guard uses pure dtype-string check (`str(a.dtype) in ("fp64", "f64", "float64")`) — no Triton dtype objects, no CUDA imports. Preserves backend-agnostic layering constraint.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness
- `compiler.py` and `_semantic.py` are clean for the inference hook work in Plan 02
- No silent f64→Fp32 coercion remains anywhere in the pipeline
- Ready for `01-02-PLAN.md` (Inference Hook & Single-Parse Seam)

## Verification Results

```
grep -c 'f64.*Fp32\|fp64.*Fp32' compiler.py → 0  ✓
grep -c 'del llvm_mod' compiler.py            → 1  ✓
grep -c 'float64' _semantic.py                → 2  ✓
pytest python/test/gluon/test_extern_call.py  → 4 passed ✓
```

---
*Phase: 01-seam-cleanup*
*Completed: 2026-07-11*
