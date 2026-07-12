---
phase: 02-semantic-time-inference
plan: 01
subsystem: compiler
tags: [cuda, clang, template-deduction, return-type-inference]

requires:
  - phase: 01-seam-cleanup
    provides: InferExternCallResult hook via codegen_fns, suspended CUDACompiler with parse-counter guard

provides:
  - PlaceholderLayout struct in tt_plugin.cu enabling layout-independent dtype+shape template deduction
  - BuildTensor placeholder-mode branch (D-07)
  - CUDACompiler::inferReturnTypes inference-only method (D-01)
  - Tensor implicit conversion from PlaceholderLayout to any concrete Layout

affects:
  - 02-02 (Python bindings + hook consumes inferReturnTypes)
  - 02-03 (semantic-time consumption via call_extern)

tech-stack:
  added: []
  patterns:
    - Implicit conversion constructor pattern for layout-independent template deduction
    - Inference-only compiler method sharing coroutine infrastructure with full codegen path

key-files:
  created: []
  modified:
    - python/test/gluon/tt_plugin.cu - PlaceholderLayout struct + Tensor implicit conversion constructor
    - python/src/clang_compiler.h - inferReturnTypes declaration in CUDACompiler struct
    - python/src/clang_compiler.cc - BuildTensor placeholder branch + inferReturnTypes method body

key-decisions:
  - "Used PlaceholderLayout with implicit conversion for dtype+shape-only inference (D-05/D-06)"
  - "PlaceholderLayout lookup failure falls through to existing concrete-layout code path for backward compatibility"

requirements-completed: [INFER-01, INFER-02]

coverage:
  - id: D1
    description: "PlaceholderLayout struct + Tensor implicit conversion constructor in tt_plugin.cu"
    requirement: INFER-01
    verification:
      - kind: unit
        ref: "grep -c 'struct PlaceholderLayout' python/test/gluon/tt_plugin.cu → 1"
        status: pass
      - kind: unit
        ref: "grep -c 'Tensor(const Tensor<T2, TShape2, PlaceholderLayout>' python/test/gluon/tt_plugin.cu → 1"
        status: pass
    human_judgment: false
  - id: D2
    description: "BuildTensor placeholder-mode branch in clang_compiler.cc (D-07)"
    requirement: INFER-01
    verification:
      - kind: unit
        ref: "grep -c 'N_WARPS == 0' python/src/clang_compiler.cc → 1"
        status: pass
      - kind: unit
        ref: "grep -c 'PlaceholderLayout' python/src/clang_compiler.cc → 4"
        status: pass
    human_judgment: false
  - id: D3
    description: "CUDACompiler::inferReturnTypes inference-only method (D-01)"
    requirement: INFER-02
    verification:
      - kind: unit
        ref: "grep -c 'CUDACompiler::inferReturnTypes' python/src/clang_compiler.cc → 1"
        status: pass
      - kind: unit
        ref: "grep -c 'inferReturnTypes' python/src/clang_compiler.h → 1"
        status: pass
      - kind: unit
        ref: "awk '/CUDACompiler::inferReturnTypes/,/^}/' clang_compiler.cc | grep -c 'InstantiationFunction\\|EmitFinalModule' → 0 (no codegen)"
        status: pass
    human_judgment: false

duration: 2min
completed: 2026-07-11
status: complete
---

# Phase 02 Plan 01: Device Library + C++ Inference Core

**PlaceholderLayout in tt_plugin.cu + BuildTensor placeholder-mode branch + CUDACompiler::inferReturnTypes inference-only method**

## Performance

- **Duration:** 2 min
- **Started:** 2026-07-11T12:09:47Z
- **Completed:** 2026-07-11T12:12:17Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `PlaceholderLayout` struct and `Tensor` implicit conversion constructor in `tt_plugin.cu`, enabling layout-independent dtype+shape template deduction (D-05/D-06)
- Added placeholder-detection branch to `CUDACompiler::BuildTensor`: when all bases are empty and `N_WARPS==0`, builds `Tensor<T,Shape,PlaceholderLayout>` via AST lookup instead of a concrete `Layout<REGS,LANES,WARPS>` (D-07)
- Added `CUDACompiler::inferReturnTypes` — an inference-only method that runs Phase 1 of `compileBitcode` (BuildTensor → LookupFunction → EvaluateFunctionReturnType) without emitting LLVM bitcode, returning per-result `TensorParameter` data (D-01)
- Declared `inferReturnTypes` in `clang_compiler.h` alongside the existing `compileBitcode`
- dtype/shape hard-errors in `tritonPatchExternCallResultTypes` preserved as safety asserts (D-10)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add PlaceholderLayout + implicit conversion to tt_plugin.cu** - `c6fe93d011` (feat)
2. **Task 2: BuildTensor placeholder mode + inferReturnTypes in clang_compiler.cc/h** - `66eb4a99c4` (feat)

## Files Modified

- `python/test/gluon/tt_plugin.cu` — Added `struct PlaceholderLayout {}` after `Layout` template; added implicit conversion constructor `Tensor(const Tensor<T2,TShape2,PlaceholderLayout>&)` inside `Tensor` struct
- `python/src/clang_compiler.h` — Added `inferReturnTypes` method declaration to `CUDACompiler` struct (after `compileBitcode`, before closing `};`)
- `python/src/clang_compiler.cc` — Added placeholder-detection branch in `BuildTensor` (early return when all bases empty + `N_WARPS==0`, building `Tensor<T,Shape,PlaceholderLayout>`); added `inferReturnTypes` method body between `EmitFinalModule` and `compileBitcode`

## Decisions Made

- Used `PlaceholderLayout::empty()` with `llvm::dyn_cast<RecordDecl>` + `static_cast<const TypeDecl*>` for AST lookup due to LLVM's deleted `getTypeDeclType(TagDecl*)` overload
- `BuildTensor` placeholder branch returns from the lambda on success, skipping concrete layout building — lookup failure falls through to existing path for backward compatibility
- `inferReturnTypes` omits `InitializeNVPTXBackend()` (no codegen emitted) and the multi-return extractor lookup (not needed at inference time)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed getTypeDeclType API change in clang/LLVM 19+**

- **Found during:** Task 2 (BuildTensor placeholder branch)
- **Issue:** `Ctx.getTypeDeclType(const TagDecl*)` was deleted in LLVM 19+ due to ambiguity with derived-type overloads. `RecordDecl*` also selected the deleted overload. `getRecordType` and `getTagDeclType` do not exist.
- **Fix:** Explicit upcast via `static_cast<const clang::TypeDecl*>(RD)` forces overload resolution to select the base `TypeDecl*` overload at `ASTContext.h:1880`.
- **Files modified:** `python/src/clang_compiler.cc` (BuildTensor lambda)
- **Verification:** LSP diagnostics cleared; existing pattern `getTypeDeclType(TypeDecl*)` is the canonical API
- **Committed in:** `66eb4a99c4` (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minor API adaptation for clang diagnostic change — no behavioral impact.

## Issues Encountered

None — plan executed smoothly.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Build integration deferred to Plan 02-03 (full build + test after all plans complete)
- Ready for Plan 02-02: Python bindings (`SuspendedCudaCompiler.infer()`) + hook (`InferExternCallResult.infer_result()`) consumption of the new C++ inference method
- The `PlaceholderLayout` and `inferReturnTypes` are available for Plan 02-02 consumers

---
*Phase: 02-semantic-time-inference*
*Completed: 2026-07-11*
