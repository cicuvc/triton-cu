# Phase 3: Verification - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-12
**Phase:** 3-verification
**Areas discussed:** New test's change dimension, New CUDA device function design, TEST-03 verification method, Precision / dtype matrix

---

## Area Selection

Presented 4 gray areas; user selected: Precision / dtype matrix, TEST-03 verification method, New test's change dimension. (New CUDA device function design followed naturally from the change-dimension choice.)

---

## New Test's Change Dimension

| Option | Description | Selected |
|--------|-------------|----------|
| Shape AND dtype together | One test changing both (f16 [32,32] → f32 [32] reduction). Exercises INFER-01 + INFER-02 together. | ✓ |
| Dtype only | Change only dtype, same shape (f16 [512] → f32 [512]). Isolates INFER-02. | |
| Two separate tests | dtype-only + shape+dtype for independent coverage. | |

**User's choice:** Shape AND dtype together.
**Notes:** dtype path (INFER-02) has zero E2E coverage today; combining shape+dtype in one test is the strongest single proof.

---

## New CUDA Device Function Design

| Option | Description | Selected |
|--------|-------------|----------|
| f16→f32 reduction | `Tensor<half,Shape<32,32>>` → `Tensor<float,Shape<32>>`, accumulate in f32. Mirrors existing `reduce`. Ref `x.to(float32).sum(1)`. | ✓ |
| bf16→f32 reduction | Same shape change, bf16 input. Needs cc>=80 gate. | |
| int32→int64 reduction | Integer accumulation, exact, no tolerance. | |

**User's choice:** f16→f32 reduction.
**Notes:** Reuses `reduce` structure and `TArg`/`TRes` layouts; only input/return element types and accumulator dtype change.

---

## TEST-03 Verification Method

| Option | Description | Selected |
|--------|-------------|----------|
| Implicit via successful run | Successful compile + correct GPU result is sufficient (internal llvm.verify_module aborts on failure); confirm no lit files touched. | |
| Explicit verify + run lit | Assert llvm.verify_module / inferred type explicitly AND run make test-lit as a regression gate. | ✓ |

**User's choice:** Explicit verify + run lit.
**Follow-up — how to prove verification:**

| Option | Description | Selected |
|--------|-------------|----------|
| Assert inferred result type in test | Inspect compiled artifact (e.g. kernel.asm['ttgir']) to confirm f32 + [32] extern_call result type, plus run make test-lit. | ✓ |
| Rely on pipeline verify + run lit | Successful compile as proof, still run make test-lit separately. | |

**Notes:** Test must prove inference produced the right type (f32 + [32]) before lowering, not just that the number is correct. `make test-lit` runs as a documented separate regression gate.

---

## Precision / dtype Matrix

| Option | Description | Selected |
|--------|-------------|----------|
| rtol=1e-2, atol=1e-2 | Standard f16-input reduction tolerance; f32 accumulation keeps error low. | ✓ |
| rtol=1e-1 | Looser bound against f16 rounding. | |
| assert_close defaults | Use dtype-based auto-tolerances. | |

**User's choice:** rtol=1e-2, atol=1e-2.
**Notes:** Only the single f16→f32 transition is exercised; bf16 and int variants deferred.

---

## the agent's Discretion

- Names of the new device function, kernel, and test.
- Reuse existing `TArg`/`TRes` aliases vs parallel half-typed aliases.
- Exact API to inspect for the inferred-type assertion (`kernel.asm['ttgir']` text match vs structured metadata).
- Assertion message wording.

## Deferred Ideas

- bf16→f32 reduction test (no new coverage beyond f16→f32 this milestone).
- int32→int64 reduction test (int dtype path; beyond single-test scope).
- Two separate tests (dtype-only + shape+dtype).
- AUTO-01 (optional result_layout), FP64-01 (full Fp64) — pre-existing deferrals.
