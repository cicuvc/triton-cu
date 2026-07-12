---
phase: 02-semantic-time-inference
plan: 05
subsystem: gluon
tags: [cuda, gluon, extern-call, fail-loud, error-handling, non-cuda-backend]

requires:
  - phase: 02-semantic-time-inference
    plan: 03
    provides: call_extern hook consumption, first_input fallback path
  - phase: 02-semantic-time-inference
    plan: 04
    provides: try/except RuntimeError removal, flat inference block in call_extern

provides:
  - call_extern raises RuntimeError when infer_extern_call_result hook is absent (non-CUDA backend)
  - Automated test verifying the hook-absent raise without silent fallback

affects:
  - Phase 02 verification (closes Gap 2 / PLAN 02-03 must_have FAILED)

tech-stack:
  added: []
  patterns:
    - Fail-loud else-branch on inference-hook presence check
    - Monkeypatched make_ir to simulate hook-absent (non-CUDA) backend in test

key-files:
  created: []
  modified:
    - python/triton/experimental/gluon/language/_semantic.py - else-branch raise when infer_hook is None
    - python/test/gluon/test_extern_call.py - test_gl_call_no_inference_hook_raises + unittest.mock.patch import

key-decisions:
  - "Raise RuntimeError (fail-loud) rather than graceful degrade — gl.call() is inherently a CUDA operation; non-CUDA backends cannot execute it. Code paths that never call gl.call() are unaffected."
  - "Test matches triton.compiler.errors.CompilationError (the wrapper) rather than bare RuntimeError, since the JIT compile pipeline wraps call_extern errors."
  - "Test patches GluonASTSource.make_ir to strip the inference hook, simulating a non-CUDA backend on CUDA hardware."

patterns-established:
  - "Hook-presence guard: if infer_hook is not None → inference path; else → explicit RuntimeError (no silent first_input fallback for missing hook)"

requirements-completed: [INFER-06]

coverage:
  - id: D1
    description: "call_extern raises RuntimeError with exact message when infer_extern_call_result hook is absent"
    requirement: INFER-06
    verification:
      - kind: unit
        ref: "grep -q 'gl.call() extern CUDA calls require the CUDA backend' python/triton/experimental/gluon/language/_semantic.py → exit 0"
        status: pass
      - kind: unit
        ref: "else-branch raise present at _semantic.py:279-283 after 'if infer_hook is not None:'"
        status: pass
    human_judgment: false
  - id: D2
    description: "Automated test verifies the hook-absent raise with specific error-message match"
    requirement: INFER-06
    verification:
      - kind: unit
        ref: "test_gl_call_no_inference_hook_raises present in python/test/gluon/test_extern_call.py"
        status: pass
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py::test_gl_call_no_inference_hook_raises (CUDA/GPU required)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Existing 4 extern-call tests still pass (CUDA backend always provides the hook)"
    requirement: INFER-06
    verification:
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py — 5 passed (4 existing + 1 new)"
        status: pass
    human_judgment: false

completed: 2026-07-11
status: complete
---

# Phase 02 Plan 05: Gap 2 Closure — Fail-Loud on Absent Inference Hook Summary

**`call_extern` now raises a clear `RuntimeError` when the `infer_extern_call_result` hook is absent (non-CUDA backend) instead of silently falling through to the `first_input`-based fallback, with an automated test verifying the raise.**

## Accomplishments

- Added an `else:` branch on the `if infer_hook is not None:` guard in `call_extern` (`_semantic.py:279-283`) that raises `RuntimeError("gl.call() extern CUDA calls require the CUDA backend. No inference hook (infer_extern_call_result) found in codegen_fns.")`.
- The raise terminates execution before reaching the downstream `first_input` fallback, so a missing hook can no longer produce a silently-wrong result type. The downstream `else` fallback (for `inferred_results is None`) is preserved as defense-in-depth for the fixed-layout-function path.
- Added `test_gl_call_no_inference_hook_raises`, which monkeypatches `GluonASTSource.make_ir` to strip the inference hook from `codegen_fns` (simulating a non-CUDA backend on CUDA hardware) and asserts the expected error surfaces.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add hook-absent error raise in call_extern** — `bd62013fd8` (feat)
2. **Task 2: Add automated test verifying hook-absent raise** — `4f47f98f8a` (test)

## Files Modified

- `python/triton/experimental/gluon/language/_semantic.py` — 5-line `else:` branch raising `RuntimeError` with the exact contract message when `infer_hook is None`.
- `python/test/gluon/test_extern_call.py` — `from unittest.mock import patch` import + `test_gl_call_no_inference_hook_raises` (31 lines) using `patch.object(_rt.GluonASTSource, 'make_ir', ...)`.

## Deviations from Plan

**1. [Rule 1] Patched `GluonASTSource.make_ir` instead of `GluonJITFunction.make_ir`**
- The plan pseudo-code patched `GluonJITFunction.make_ir`; the actual make_ir entry point in this codebase is `GluonASTSource.make_ir`. Used the real symbol.

**2. [Rule 1] Test matches `triton.compiler.errors.CompilationError` rather than bare `RuntimeError`**
- The Gluon JIT compile pipeline wraps errors raised inside `call_extern` in a `CompilationError`. `pytest.raises` therefore targets the wrapper with the same message regex (`r"gl\.call\(\) extern CUDA calls require the CUDA backend"`), which still asserts the contract message is surfaced to the user.

**Total deviations:** 2 auto-fixed (Rule 1). Both preserve the plan's intent (verify the fail-loud raise reaches the user with the correct message).

## Issues Encountered

- Executor session was interrupted while writing this SUMMARY.md after both task commits landed; the orchestrator finalized the summary and tracking. Implementation and tests were fully committed prior to the interruption.

## User Setup Required

None.

## Next Phase Readiness

- Gap 2 is closed. Both gaps from `02-VERIFICATION.md` (Gap 1 in 02-04, Gap 2 here) are now resolved.
- Phase 02 has no remaining incomplete plans — ready for phase verification.

---

*Phase: 02-semantic-time-inference*
*Completed: 2026-07-11*
