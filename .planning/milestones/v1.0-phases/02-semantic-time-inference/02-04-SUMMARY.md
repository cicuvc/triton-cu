---
phase: 02-semantic-time-inference
plan: 04
subsystem: compiler
tags: [cuda, clang, template-deduction, return-type-inference, gluon, placeholder-layout, fallback]

requires:
  - phase: 02-semantic-time-inference
    plan: 01
    provides: PlaceholderLayout struct, BuildTensor placeholder branch, CUDACompiler::inferReturnTypes
  - phase: 02-semantic-time-inference
    plan: 02
    provides: SuspendedCudaCompiler.infer() pybind11 binding, InferExternCallResult.infer_result() hook
  - phase: 02-semantic-time-inference
    plan: 03
    provides: call_extern hook consumption, first_input fallback for fixed-layout functions

provides:
  - LookupFunctionWithPlaceholderFallback C++ method for resolving fixed-layout function signatures
  - All extern-call functions (including reduce with TArg/TRes) obtain return dtype+shape from CUDA inference
  - try/except RuntimeError workaround removed — CUDA inference failures are no longer silently swallowed

affects:
  - 02-05 (shape/dtype propagation through result type)
  - Phase 03 or future: auto-layout resolution

tech-stack:
  added: []
  patterns:
    - PlaceholderLayout + ExplicitTemplateArgs fallback for fixed-layout template deduction
    - Coroutine-based C++ fallback method following existing TaskQueue/SwitchTo pattern
    - DeduceTemplateArguments with explicit TemplateArgumentListInfo enabling PerformCopyInitialization

key-files:
  created: []
  modified:
    - python/src/clang_compiler.h - LookupFunctionWithPlaceholderFallback declaration with Mechanism (a) decision comment
    - python/src/clang_compiler.cc - LookupFunctionWithPlaceholderFallback implementation + inferReturnTypes integration
    - python/triton/experimental/gluon/language/_semantic.py - Removed try/except RuntimeError catch; inference now succeeds for all functions

key-decisions:
  - "Chose Mechanism (a) — PlaceholderLayout + ExplicitTemplateArgs over direct instantiation because it reuses existing code (PlaceholderLayout struct, implicit conversion ctor, BuildTensor placeholder branch)"
  - "Used Ctx.getTrivialTypeSourceInfo(elementQualType) for TemplateArgumentLoc construction instead of default TemplateArgumentLocInfo() — the default ctor crashes DeduceTemplateArguments"
  - "Replaced dummy-concrete-bases approach with PlaceholderLayout arg types because dummy bases don't match fixed (non-template) layout parameters like TArg/TRes"

patterns-established:
  - "Fallback-on-LookupFailure pattern: primary LookupFunction (dummy bases) → LookupFunctionWithPlaceholderFallback (PlaceholderLayout + explicit args) → error return"
  - "PlaceholderLayout BuildTensor with instantiate=false avoid clang Sema crashes while still producing valid QualType for template argument deduction"

requirements-completed: [INFER-01, INFER-02, INFER-03]

coverage:
  - id: D1
    description: "LookupFunctionWithPlaceholderFallback resolves fixed-layout function signatures via PlaceholderLayout + explicit template args"
    requirement: INFER-01
    verification:
      - kind: unit
        ref: "grep -c 'LookupFunctionWithPlaceholderFallback' python/src/clang_compiler.cc → 2"
        status: pass
      - kind: unit
        ref: "grep -c 'LookupFunctionWithPlaceholderFallback' python/src/clang_compiler.h → 1"
        status: pass
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py::test_reduce_different_shape - PASSED"
        status: pass
    human_judgment: false
  - id: D2
    description: "All 4 extern-call tests pass with CUDA inference for every function (no first_input fallback for reduce)"
    requirement: INFER-02
    verification:
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py - 4 passed"
        status: pass
    human_judgment: false
  - id: D3
    description: "try/except RuntimeError catch removed; CUDA inference failures now propagate as errors"
    requirement: INFER-03
    verification:
      - kind: unit
        ref: "grep -c 'except RuntimeError' python/triton/experimental/gluon/language/_semantic.py → 0"
        status: pass
      - kind: unit
        ref: "grep -c '# Fallback: infer from first_input' python/triton/experimental/gluon/language/_semantic.py → 1 (preserved for non-CUDA backends)"
        status: pass
    human_judgment: false

duration: 18min
completed: 2026-07-11
status: complete
---

# Phase 02 Plan 04: Gap 1 Closure — Fixed-Layout Function Inference Summary

**Fixed-layout `reduce` (and any function with concrete TArg/TRes layout params) obtains return dtype+shape from CUDA inference via PlaceholderLayout + ExplicitTemplateArgs fallback, removing the try/except RuntimeError workaround**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-07-11T14:10:00Z
- **Completed:** 2026-07-11T14:28:00Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `LookupFunctionWithPlaceholderFallback` implemented in `clang_compiler.cc` — when `LookupFunction` returns nullptr for fixed-layout functions, the fallback rebuilds arg types using `PlaceholderLayout` (N_WARPS=0, empty bases, `instantiate=false`) and constructs explicit `TemplateArgumentListInfo` with the element type, enabling `DeduceTemplateArguments` to treat the function as non-template for argument checking (using `PerformCopyInitialization` with implicit conversions)
- The fallback succeeds for `reduce` (and any function whose template parameters are satisfied by the element type alone) because the implicit `Tensor(PlaceholderLayout)→Tensor(ConcreteLayout)` conversion constructor in `tt_plugin.cu:91-98` handles the layout parameter matching
- `inferReturnTypes` now tries the fallback when `LookupFunction` fails, before returning the error — only `inferReturnTypes` is modified; `compileBitcode` continues using the primary lookup path
- Removed `try/except RuntimeError` wrapper at `_semantic.py:282-283` that silently swallowed CUDA inference failures — CUDA inference failures now propagate as real errors
- The `else:` first_input fallback path at `_semantic.py:297-303` is preserved for non-CUDA backends where the `infer_hook` is absent
- All 4 existing extern-call tests pass unchanged: `elementwise_add`, `intra_warp_add_sibling`, `reduce_different_shape`, `split_add_tuple`

## Task Commits

Each task was committed atomically:

1. **Task 1: Investigate & Reproduce + Decision** - `112e6dfe91` (feat)
2. **Task 2: Implement LookupFunctionWithPlaceholderFallback** - `88772a10bb` (feat)
3. **Task 3: Remove try/except RuntimeError in _semantic.py** - `5aa4250da9` (feat)

## Files Modified

- `python/src/clang_compiler.h` — New method declaration `LookupFunctionWithPlaceholderFallback` with detailed Mechanism (a) decision comment explaining why PlaceholderLayout + explicit template args succeeds where dummy concrete bases fail
- `python/src/clang_compiler.cc` — New method body (97 lines) between `LookupFunction` and `InstantiationFunction`: builds PlaceholderLayout arg types, constructs explicit `TemplateArgumentListInfo` via `getTrivialTypeSourceInfo`, calls `DeduceTemplateArguments` with explicit args; `inferReturnTypes` integration at line 965-970 (tries fallback when primary lookup fails)
- `python/triton/experimental/gluon/language/_semantic.py` — Removed `try:/except RuntimeError:` wrapper at lines 272-283; de-dented the `arg_params`/`inferred_results` block; preserved `else:` first_input fallback for non-CUDA backends

## Decisions Made

- Selected Mechanism (a) — PlaceholderLayout + ExplicitTemplateArgs — over Mechanism (b) — Direct Instantiation. Rationale: reuses existing code (PlaceholderLayout struct, implicit conversion constructor, BuildTensor placeholder branch); the only new code is the fallback method wiring. Direct instantiation would require fragile per-function template param matching by position/heuristic.
- Used `Ctx.getTrivialTypeSourceInfo(elementQualType)` for `TemplateArgumentLoc` construction — the default `TemplateArgumentLocInfo()` constructor causes `DeduceTemplateArguments` to abort (likely an assertion on null source info for type args)
- Set `SL` to a valid source location via `SemaRef.getSourceManager().getLocForStartOfFile()` instead of default-constructed `SourceLocation()` to avoid potential assertion failures in OpaqueValueExpr

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] TemplateArgumentLocInfo default constructor causes SIGABRT in DeduceTemplateArguments**
- **Found during:** Task 2 (implementation and testing)
- **Issue:** Using `clang::TemplateArgumentLocInfo()` (default constructor) to build the `TemplateArgumentLoc` caused `DeduceTemplateArguments` to abort with a fatal Python error. The default constructor sets the TypeSourceInfo pointer to null, which triggers an assertion in clang's template argument handling.
- **Fix:** Used `Ctx.getTrivialTypeSourceInfo(elementQualType)` to construct a valid `TypeSourceInfo*` for the type template argument, then passed it to `TemplateArgumentLoc(TArg, TSI)`.
- **Files modified:** `python/src/clang_compiler.cc` (Fallback method)
- **Committed in:** `88772a10bb` (part of Task 2 commit)

**2. [Rule 1 - Bug] null SourceLocation may cause assertion failures**
- **Found during:** Task 2 (investigation)
- **Issue:** Default-constructed `clang::SourceLocation()` is invalid and may trigger assertions in `OpaqueValueExpr` or `DeduceTemplateArguments` when used as a source location.
- **Fix:** Set `SL` to `SemaRef.getSourceManager().getLocForStartOfFile(SemaRef.getSourceManager().getMainFileID())`, matching the pattern used by `FunctionResolver`.
- **Files modified:** `python/src/clang_compiler.cc` (Fallback method)
- **Committed in:** `88772a10bb` (part of Task 2 commit)

**3. [Rule 3 - Blocking] Edit tool matched multiple identical code blocks**
- **Found during:** Task 1 and Task 2 (code editing)
- **Issue:** `inferReturnTypes` and `compileBitcode` have nearly identical `LookupFunction` call-site code, causing the Edit tool to modify both locations when targeting only `inferReturnTypes`.
- **Fix:** Used larger unique context strings (including method-specific comments like "Phase 1: Type inference only") to disambiguate the edit targets. In one case, reverted unintended changes and used `sed`-based approach.
- **Files modified:** `python/src/clang_compiler.cc` (multiple cycles of fix/revert)
- **Committed in:** N/A (corrected inline during implementation)

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bugs, 1 Rule 3 blocking)
**Impact on plan:** All fixes necessary for correctness. The main fix (getTrivialTypeSourceInfo) was critical — without it, the C++ fallback crashes and the entire Gap 1 closure would fail.

## Issues Encountered

- The `DeduceTemplateArguments` with `nullptr` explicit args (no explicit template args) returns `NonDeducedMismatch` for `reduce` even with PlaceholderLayout arg types — confirming that explicit template args are necessary. When explicit args are provided (via `&TALI`), clang uses `PerformCopyInitialization` which tries implicit conversions, and the `Tensor(PlaceholderLayout)→Tensor(ConcreteLayout)` conversion constructor matches.
- The `edit` tool matched code blocks in both `inferReturnTypes` and `compileBitcode` because of structural similarity — required careful use of unique surrounding context to achieve single-location edits.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Gap 1 is closed. `reduce` (and any fixed-layout function) now obtains return dtype+shape from CUDA inference.
- Ready for Plan 02-05: propagate CUDA-inferred shape/dtype through the result type construction pipeline and verify with an E2E test.
- All 4 existing tests still pass — no regressions.

---

*Phase: 02-semantic-time-inference*
*Completed: 2026-07-11*
