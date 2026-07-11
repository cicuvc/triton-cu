---
phase: 03-verification
plan: 01
subsystem: testing
tags: [cuda, gluon, gl.call, return-type-inference, f16, f32, reduction, e2e-test]

requires:
  - phase: 02-semantic-time-inference
    provides: "CUDA-inferred return dtype+shape flowing into ttg.extern_call result type at semantic time"
provides:
  - "E2E test exercising shape+dtype-changing gl.call() — f16→f32 reduction with inferred result type"
  - "reduce_f16 device function in tt_plugin.cu (template, PlaceholderLayout-compatible)"
  - "Regression confirmation: all 5 existing extern-call tests pass, lit suite unaffected"
affects: []

tech-stack:
  added: ["lit (Python test runner, pip install)"]
  patterns:
    - "Template device function pattern for PlaceholderLayout probe compatibility"
    - "CompiledKernel.asm['ttgir'] for inferred-type assertion in Gluon tests"

key-files:
  created: []
  modified:
    - "python/test/gluon/tt_plugin.cu — added template reduce_f16<T> device function"
    - "python/test/gluon/test_extern_call.py — added reduce_f16_kernel + test_reduce_f16_f32"

key-decisions:
  - "Made reduce_f16 a template<T> function (not concrete) to work with the PlaceholderLayout probe path in the inference pipeline — mirrors existing reduce pattern"
  - "Captured CompiledKernel return from kernel call to access .asm['ttgir'] for inferred-type assertions"

patterns-established:
  - "Template device function with PlaceholderLayout-compatible signature: template<typename T> with T used for input element type, return type fixed to concrete type"
  - "Gluon kernel inferred-type assertion: compiled = kernel[(1,)](...); ttgir = compiled.asm['ttgir']; assert text patterns in IR"

requirements-completed:
  - TEST-01
  - TEST-02
  - TEST-03

coverage:
  - id: D1
    description: "reduce_f16 device function in tt_plugin.cu — f16→f32 CUDA reduction"
    requirement: "TEST-01"
    verification:
      - kind: integration
        ref: "python/test/gluon/test_extern_call.py::test_reduce_f16_f32"
        status: pass
    human_judgment: false
  - id: D2
    description: "reduce_f16_kernel + test_reduce_f16_f32 — shape+dtype-changing gl.call() E2E test"
    requirement: "TEST-01"
    verification:
      - kind: integration
        ref: "python/test/gluon/test_extern_call.py::test_reduce_f16_f32"
        status: pass
    human_judgment: false
  - id: D3
    description: "Regression gate: all 4 existing extern-call tests still pass"
    requirement: "TEST-02"
    verification:
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py (6 tests, all pass)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Lit test suite unaffected (Gluon lit tests all pass)"
    requirement: "TEST-03"
    verification:
      - kind: other
        ref: "lit build/test/ (278 discovered, Gluon tests all pass, 125 pre-existing AMD/unrelated failures)"
        status: pass
    human_judgment: false

duration: 23 min
completed: 2026-07-11
status: complete
---

# Phase 03 Plan 01: E2E Verification — f16→f32 Reduction with Inferred Return Type

**f16→f32 CUDA reduction E2E test proving gl.call() return-type inference handles simultaneous shape AND dtype transitions — only result_layout supplied, f32 element type + [32] shape inferred from CUDA.**

## Performance

- **Duration:** 23 min
- **Started:** 2026-07-11T17:20:00Z (approx)
- **Completed:** 2026-07-11T17:43:32Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- `reduce_f16<T>` template device function in `tt_plugin.cu` — f16→f32 reduction with float accumulator, PlaceholderLayout-compatible
- `reduce_f16_kernel` + `test_reduce_f16_f32` E2E test — f16 input, f32 output, `rtol=1e-2, atol=1e-2`, inferred-type assertion on `kernel.asm['ttgir']`
- All 5 existing extern-call tests pass unchanged (6 total), lit Gluon suite unaffected

## Task Commits

1. **Task 1: Add reduce_f16 device function** — `d3719f1df9` (feat)
2. **Task 2: Add reduce_f16_kernel + test_reduce_f16_f32** — `80f045e7ec` (feat, includes template fix discovered during verification)
3. **Task 3: Regression gates** — no file changes (verification-only)

## Files Created/Modified

- `python/test/gluon/tt_plugin.cu` — Added `template<typename T> __device__ Tensor<float, Shape<32>, TRes> reduce_f16(...)` after the existing `reduce`, before `split_add`. Uses `TArg`/`TRes` layout aliases, float accumulator, half→float implicit promotion via `+=`.
- `python/test/gluon/test_extern_call.py` — Added `reduce_f16_kernel` (after `split_add_kernel`) and `test_reduce_f16_f32` (after `test_split_add_tuple`, before `test_gl_call_no_inference_hook_raises`). Test checks numeric correctness and asserts `f32` + `tensor<32xf32` in compiled ttgir.

## Verification Results

| Gate | Command | Result |
|------|---------|--------|
| New test isolation | `pytest test_extern_call.py::test_reduce_f16_f32` | **PASSED** — f16→f32 numeric + inferred-type assertions |
| Full extern-call suite | `pytest test_extern_call.py` (6 tests) | **PASSED** — all pass, exit 0 |
| Lit suite | `lit build/test/` (278 tests) | **PASSED** — all Gluon tests pass; 125 pre-existing AMD/unrelated failures, no new regressions |
| Test collection | `pytest --co test_extern_call.py` | **PASSED** — `test_reduce_f16_f32` collected |

## Decisions Made

- **Template vs. concrete function:** `reduce_f16` was initially written as a concrete non-template `__device__` function, but the inference pipeline's `PlaceholderLayout` fallback (`LookupFunctionWithPlaceholderFallback` → `DeduceTemplateArguments`) only resolves template functions. Changed to `template<typename T>` with `T` as the input element type, mirroring the existing `reduce<T>` pattern. Return type stays `Tensor<float, ...>` — `T` is deduced as `half` from the call site, and the return type is fixed.
- **`compiled.asm` vs. `kernel.asm`:** The `asm` dict lives on the `CompiledKernel` object returned by `JITFunction.run()`, not on the function handle itself. Captured the return value (`compiled = kernel[...](...)`) for the ttgir assertion.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] reduce_f16 found by inference pipeline**

- **Found during:** Task 2 (running test_reduce_f16_f32)
- **Issue:** `reduce_f16` was a concrete non-template function. The `PlaceholderLayout` inference probe path uses clang Sema `DeduceTemplateArguments`, which only resolves template functions. Both `LookupFunction` (canonical layout mismatch) and `LookupFunctionWithPlaceholderFallback` (PlaceholderLayout ≠ TArg) failed.
- **Fix:** Wrapped `reduce_f16` in `template<typename T>`, using `T` for the input element type. Return type stays `Tensor<float, Shape<32>, TRes>` — template deduction resolves `T=half` from the call site. This mirrors how the existing `reduce<T>` works.
- **Files modified:** `python/test/gluon/tt_plugin.cu`
- **Verification:** `pytest test_extern_call.py::test_reduce_f16_f32` passes
- **Committed in:** `80f045e7ec` (Task 2 commit)

**2. [Rule 1 - Bug] kernel.asm attribute access**

- **Found during:** Task 2 (running test_reduce_f16_f32)
- **Issue:** Test used `reduce_f16_kernel.asm["ttgir"]` but `asm` is on the `CompiledKernel` return value, not the `GluonJITFunction` object.
- **Fix:** Captured return value: `compiled = reduce_f16_kernel[(1,)](x, out, num_warps=1)` then `compiled.asm["ttgir"]`.
- **Files modified:** `python/test/gluon/test_extern_call.py`
- **Verification:** `pytest test_extern_call.py::test_reduce_f16_f32` passes with ttgir assertions
- **Committed in:** `80f045e7ec` (Task 2 commit)

**3. [Rule 3 - Blocking] lit command not found**

- **Found during:** Task 3 (Gate B — lit suite)
- **Issue:** `lit` not installed in the environment
- **Fix:** `pip install lit` (18.1.8)
- **Verification:** `lit build/test/` runs successfully
- **Committed in:** N/A (tooling install, no tracked files changed)

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bugs, 1 Rule 3 blocking)
**Impact on plan:** Both Rule 1 fixes were necessary for the test to pass. Template fix aligns with design intent (per PATTERNS.md: "deduced through [PlaceholderLayout] exactly like reduce"). No scope creep.

## Issues Encountered

- None beyond the auto-fixed deviations above.

## Known Stubs

None — device function and test are fully implemented. No placeholder values, TODOs, or mock data.

## Threat Flags

None — no new trust boundaries. Device function runs in existing CUDA kernel context; test is pure pytest. No external input, no network surface.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

Phase 03 (Verification) complete — all three requirements (TEST-01, TEST-02, TEST-03) satisfied. Ready for milestone close-out.

---

## Self-Check: PASSED

- `python/test/gluon/tt_plugin.cu` — FOUND
- `python/test/gluon/test_extern_call.py` — FOUND
- `.planning/phases/03-verification/03-01-SUMMARY.md` — FOUND
- Commit `d3719f1df9` (Task 1) — FOUND
- Commit `80f045e7ec` (Task 2) — FOUND
- Task 3 (regression gates) — 6/6 extern-call tests pass, lit Gluon suite passes

---

*Phase: 03-verification*
*Completed: 2026-07-11*
