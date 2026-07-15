---
phase: 05-mlir-op-relaxation-spec-extraction
plan: 01
subsystem: compiler-dialect
tags: [mlir, ods, tablegen, tritongpu, extern-call, anytypeof, memdesc, lit-testing]

# Dependency graph
requires: []
provides:
  - Relaxed ODS type constraint on ttg.extern_call from Variadic<TT_Tensor> to Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>
  - Lit test proving mixed tensor+memdesc operands parse successfully (SHMLIR-01)
  - Lit test proving tensor-only operands parse without regression (SHMLIR-01 regression)
affects: [06-shared-memory-lowering, any-downstream-pass-that-walks-extern-call-operands]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - AnyTypeOf<[TT_Tensor, TTG_MemDescType]> for mixed operand ODS constraints (D-09)
    - -verify-diagnostics lit test pattern for parse-only type-constraint verification (D-11)

key-files:
  created:
    - test/TritonGPU/extern-call-mixed-inputs.mlir
    - test/TritonGPU/extern-call-tensor-only.mlir
  modified:
    - include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td (line 803 only)

key-decisions:
  - "Use AnyTypeOf<[TT_Tensor, TTG_MemDescType]> for mixed operands rather than separate input lists (per D-09 decision)"
  - "Lit tests use -verify-diagnostics (parse-only) per D-11 — no FileCheck field verification needed for type constraint testing"

patterns-established:
  - "Mixed operand lists via AnyTypeOf are MLIR-idiomatic; per-operand type branching is handled by downstream passes, not the ODS constraint"
  - "Parse-only lit tests with -verify-diagnostics are sufficient for ODS constraint verification"

requirements-completed:
  - SHMLIR-01

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "ttg.extern_call ODS operand constraint relaxed to AnyTypeOf<[TT_Tensor, TTG_MemDescType]>"
    requirement: "SHMLIR-01"
    verification:
      - kind: other
        ref: "grep 'AnyTypeOf<\\[TT_Tensor, TTG_MemDescType\\]>' include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:803"
        status: pass
    human_judgment: false
  - id: D2
    description: "Mixed tensor+memdesc extern_call parses without verifier error"
    requirement: "SHMLIR-01"
    verification:
      - kind: unit
        ref: "test/TritonGPU/extern-call-mixed-inputs.mlir (lit PASS)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Tensor-only extern_call parses without regression"
    requirement: "SHMLIR-01"
    verification:
      - kind: unit
        ref: "test/TritonGPU/extern-call-tensor-only.mlir (lit PASS)"
        status: pass
    human_judgment: false

# Metrics
duration: 9min
completed: 2026-07-15
status: complete
---

# Phase 5 Plan 1: ODS Relaxation — AnyTypeOf Mixed Operand Constraint Summary

**Relaxed ttg.extern_call ODS from Variadic<TT_Tensor> to AnyTypeOf<[TT_Tensor, TTG_MemDescType]> with parse-verification lit tests (SHMLIR-01)**

## Performance

- **Duration:** 9 min
- **Started:** 2026-07-15T14:56:35Z
- **Completed:** 2026-07-15T15:05:50Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Replaced `Variadic<TT_Tensor>:$inputs` with `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs` at TritonGPUOps.td:803 (single-line change)
- Created `extern-call-tensor-only.mlir` — tensor-only regression lit test, proves AnyTypeOf does not reject pure tensor operands
- Created `extern-call-mixed-inputs.mlir` — mixed tensor+memdesc positive lit test, proves both operand types parse together
- Rebuilt triton-opt (tablegen regeneration + recompilation succeeded)
- Both new lit tests pass; full TritonGPU lit suite shows zero new failures vs. pre-change baseline
- Enables Phase 6 shared-memory lowering to accept MemDescType operands on extern_call ops (D-09 precondition met)

## Task Commits

Each task was committed atomically:

1. **Task 1: Relax ODS — AnyTypeOf at TritonGPUOps.td:803 (D-09)** - `a514d6b4fd` (feat)
   - Single-line change: `Variadic<TT_Tensor>:$inputs,` → `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs,`
   - No include/verifier/format changes needed — all pre-existing infrastructure handles it

2. **Task 2: Create lit tests — mixed-inputs + tensor-only regression (D-11, SHMLIR-01)** - `682dace4cb` (test)
   - New: `test/TritonGPU/extern-call-tensor-only.mlir` (pure-tensor parse verification)
   - New: `test/TritonGPU/extern-call-mixed-inputs.mlir` (tensor+memdesc parse verification)

3. **Task 3: Build + verify — lit test validation and regression check** - `8e882ec6cc` (fix)
   - Corrected lit test syntax and layout encoding compatibility issues discovered during verification
   - Both tests pass; full TritonGPU suite has zero new failures

**Plan metadata:** [pending final commit]

## Files Created/Modified

- `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` (line 803) — ODS operand constraint relaxed from `Variadic<TT_Tensor>` to `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>`
- `test/TritonGPU/extern-call-tensor-only.mlir` — Tensor-only regression lit test (SHMLIR-01)
- `test/TritonGPU/extern-call-mixed-inputs.mlir` — Mixed tensor+memdesc lit test (SHMLIR-01)

## Decisions Made

None — followed plan as specified. The plan's decisions (D-09: mixed operand list with AnyTypeOf; D-11: parse-only lit tests) were executed directly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed shared_linear encoding `offset = [[0, 1], [0, 2], [1, 0], [2, 0]]` creating non-bijective layout**
- **Found during:** Task 3 (lit test verification)
- **Issue:** Plan-specified shared_linear offset `[2, 0]` combined with `block = [[2, 0]]` produced a non-bijective layout, causing "After removing zero bases the layout must be bijective" verifier error
- **Fix:** Used 1-CTA shared_linear encoding from ops.mlir line 379: `offset = [[0, 1], [0, 2], [1, 0], [2, 2]]` (no block parameter), with shape `4x4xf32`
- **Files modified:** `test/TritonGPU/extern-call-mixed-inputs.mlir`
- **Verification:** `lit -v test/TritonGPU/extern-call-mixed-inputs.mlir` → PASS
- **Committed in:** `8e882ec6cc` (Task 3 commit)

**2. [Rule 1 - Bug] Fixed extern_call assembly format syntax in lit tests**
- **Found during:** Task 3 (lit test verification)
- **Issue:** Plan-specified `ttg.extern_call(%a, %b) { attrs } : (types) -> result` used wrong operand syntax. The ODS assembly format `$inputs `:` functional-type($inputs, $results) attr-dict` requires operands without parentheses and colon before attr-dict: `ttg.extern_call %a, %b : (types) -> result { attrs }`
- **Fix:** Removed parentheses around operands, moved colon before attr-dict in both test files
- **Files modified:** `test/TritonGPU/extern-call-mixed-inputs.mlir`, `test/TritonGPU/extern-call-tensor-only.mlir`
- **Verification:** Both lit tests parse correctly after fix
- **Committed in:** `8e882ec6cc` (Task 3 commit)

**3. [Rule 1 - Bug] Fixed warpsPerCTA layout mismatch in blocked encoding**
- **Found during:** Task 3 (lit test verification)
- **Issue:** Plan-specified `warpsPerCTA = [2, 2]` (4 warps) conflicted with module's `num-warps = 1`, causing verifier error "Layout has 4 warps per CTA, but the context requires 1 warps per CTA"
- **Fix:** Changed `warpsPerCTA = [1, 1]` to match module declaration
- **Files modified:** `test/TritonGPU/extern-call-tensor-only.mlir`, `test/TritonGPU/extern-call-mixed-inputs.mlir`
- **Verification:** Both lit tests pass after fix
- **Committed in:** `8e882ec6cc` (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (3 bugs in plan-specified MLIR values)
**Impact on plan:** All fixes were necessary for lit test correctness. Core ODS change unchanged. No scope creep.

## Issues Encountered

- Initial lit test failures revealed 3 plan-specified MLIR value errors (layout encoding, assembly format syntax, warpsPerCTA mismatch). All fixed inline in Task 3.
- Full TritonGPU suite runs with 71/128 pre-existing failures (AMD-specific passes not compiled, consan.mlir TritonInstrument crash) — zero attributable to this change.

## Next Phase Readiness

- **Ready for Phase 5 Plan 2** (spec extraction, if applicable) — ODS relaxation is complete and verified
- **Ready for Phase 6** (shared-memory lowering) — precondition met: `ttg.extern_call` now accepts `MemDescType` operands per D-09
- **Blast radius concern** (from STATE.md): Downstream passes assuming tensor-only inputs remain to be verified in Phase 6; the ODS `AnyTypeOf` constraint is the correct first step

---

*Phase: 05-mlir-op-relaxation-spec-extraction*
*Completed: 2026-07-15*
