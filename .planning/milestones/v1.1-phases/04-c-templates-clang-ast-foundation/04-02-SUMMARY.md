---
phase: 04-c-templates-clang-ast-foundation
plan: 02
subsystem: compiler
tags: [cuda, clang-ast, shared-memory, template-specialization, nttp, type-builder, type-inspector]

requires:
  - phase: 04-c-templates-clang-ast-foundation
    plan: 01
    provides: SharedLinearLayout/SharedTensor device templates, SharedLayoutInfo/SharedTensorParameter structs, TypeBuilder/TypeInspector extended declarations, SharedTensorParameter pybind11 binding
provides:
  - TypeBuilder::BuildSharedLinearLayout (forward AST construction: SharedLayoutInfo → SharedLinearLayout<OB, BB, Align> QualType)
  - TypeBuilder::BuildSharedTensor (forward: scalar+shape+layout → SharedTensor<T,Shape,SharedLinearLayout> QualType)
  - TypeInspector::ParseSharedBasis (reverse: OffsetBases/BlockBases NTTP → flat basis vectors)
  - TypeInspector::ParseSharedTensorType (reverse: SharedTensor<...> → SharedTensorParameter)
  - TypeInspector::DispatchTypeParsing extended with SharedTensorTemplateType branch
  - CUDACompiler::BuildSharedTensor (TaskQueue coroutine pattern for shared-tensor params)
  - compileBitcode/inferReturnTypes variant dispatch: std::get_if<SharedTensorParameter>
affects: [04-c-templates-clang-ast-foundation-plan-03, 05-ods-relaxation, 06-mlir-lowering]

tech-stack:
  added: []
  patterns:
    - OffsetBases/BlockBases NTTP carriers constructed via BuildBasisGroup-style APValue pattern (struct{array{struct{array{int32}}}})
    - SharedLinearLayout specialization: 3 template args (OffsetBases NTTP, BlockBases NTTP, Alignment integral)
    - SharedTensor specialization: 3 template args (ElementType, ShapeType, LayoutType) — mirror of BuildTensor but no instantiation
    - ParseSharedBasis: walks 3-level APValue nest (carrier struct → array → IntTuple struct → array → int32)
    - SharedTensorParameter variant dispatch between TensorParameter and ScalarType arms in compileBitcode/inferReturnTypes

key-files:
  created: []
  modified:
    - python/src/clang_compiler.cc — TypeBuilder::BuildSharedLinearLayout + BuildSharedTensor, TypeInspector::ParseSharedBasis + ParseSharedTensorType + DispatchTypeParsing extension, CUDACompiler::BuildSharedTensor + variant dispatch
    - python/src/clang_compiler.h — ParseSharedBasis declaration, CUDACompiler::BuildSharedTensor declaration

key-decisions:
  - "BuildSharedLinearLayout uses lambda-based NTTP carrier construction — builds OffsetBases/BlockBases specializations with IntTuple<RANK> APValues, then SharedLinearLayout from resulting NTTPs"
  - "ParseSharedBasis is a standalone method (not a parameter to ParseBasis) because OffsetBases/BlockBases carriers have a different template parent (top-level, not LayoutFactory-nested)"
  - "DispatchTypeParsing places SharedTensorTemplateType check BEFORE TensorTemplateType to avoid ambiguity"

patterns-established:
  - "NTTP carrier construction lambda pattern: lookup template decl → create specialization → build APValue → getTemplateParamObjectDecl → return {TemplateArgument, Spec} pair"
  - "ParseSharedBasis 3-level APValue walk: Struct{0} → Array → Struct{0} → Array → getInt()"

requirements-completed:
  - SHAST-02
  - SHAST-03

coverage:
  - id: D1
    description: "TypeBuilder::BuildSharedLinearLayout constructs SharedLinearLayout<OffsetBases{...}, BlockBases{...}, Align> clang AST from SharedLayoutInfo"
    requirement: SHAST-02
    verification:
      - kind: unit
        ref: "CC=clang CXX=clang++ bash build.sh — clang_compiler.cc compiles"
        status: pass
    human_judgment: false
  - id: D2
    description: "TypeBuilder::BuildSharedTensor constructs SharedTensor<T, Shape<N>, SharedLinearLayout<...>> clang AST"
    requirement: SHAST-02
    verification:
      - kind: unit
        ref: "CC=clang CXX=clang++ bash build.sh — clang_compiler.cc compiles"
        status: pass
    human_judgment: false
  - id: D3
    description: "TypeInspector::ParseSharedTensorType + DispatchTypeParsing shared-tensor branch; ParseSharedBasis helper for flat basis extraction"
    requirement: SHAST-03
    verification:
      - kind: unit
        ref: "CC=clang CXX=clang++ bash build.sh — clang_compiler.cc compiles"
        status: pass
    human_judgment: false
  - id: D4
    description: "CUDACompiler::BuildSharedTensor with TaskQueue coroutine pattern, compileBitcode/inferReturnTypes variant dispatch for SharedTensorParameter"
    requirement: SHAST-03
    verification:
      - kind: unit
        ref: "CC=clang CXX=clang++ bash build.sh — clang_compiler.cc compiles"
        status: pass
      - kind: unit
        ref: "PYTHONPATH smoke: CudaFuncRequest().param_types = [SharedTensorParameter()] prints OK"
        status: pass
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py — 6/6 pass (zero regression)"
        status: pass
    human_judgment: false

duration: 7min
completed: 2026-07-12
status: complete
---

# Phase 04 Plan 02: TypeBuilder, TypeInspector & CUDACompiler Shared-Tensor Path Summary

**Complete forward (TypeBuilder) and reverse (TypeInspector) clang AST round-trip for SharedTensor types, and integrate SharedTensorParameter into the CUDACompiler variant-dispatch paths**

## Performance

- **Duration:** 7 min
- **Started:** 2026-07-12T14:36:28Z
- **Completed:** 2026-07-12T14:42:33Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- TypeBuilder::BuildSharedLinearLayout constructs SharedLinearLayout<OffsetBases{...}, BlockBases{...}, Align> from SharedLayoutInfo — complete forward AST construction
- TypeBuilder::BuildSharedTensor constructs SharedTensor<T, Shape<N>, SharedLinearLayout<...>> mirroring BuildTensor's specialization pattern
- TypeInspector::ParseSharedBasis extracts flat basis vectors from OffsetBases/BlockBases NTTP carriers via 3-level APValue walk
- TypeInspector::ParseSharedTensorType parses SharedTensor<...> back to SharedTensorParameter with scalar type, shape dims, offset/block bases, and alignment
- DispatchTypeParsing dispatches SharedTensor types before TensorType to avoid ambiguity
- CUDACompiler::BuildSharedTensor uses TaskQueue coroutine pattern (BuildSharedLinearLayout → BuildSharedTensor)
- compileBitcode and inferReturnTypes variant-dispatch loops handle SharedTensorParameter via std::get_if

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement TypeBuilder::BuildSharedLinearLayout and BuildSharedTensor** — `a254aa4326` (feat)
2. **Task 2: Implement TypeInspector::ParseSharedTensorType + DispatchTypeParsing extension** — `3908028eec` (feat)
3. **Task 3: Add CUDACompiler::BuildSharedTensor + variant dispatch in compileBitcode/inferReturnTypes** — `5ce6da0d85` (feat)

## Files Modified
- `python/src/clang_compiler.cc` — TypeBuilder constructor (2 new initializers), BuildSharedLinearLayout (130 lines), BuildSharedTensor (22 lines), ParseSharedBasis (24 lines), ParseSharedTensorType (30 lines), DispatchTypeParsing (5 lines), CUDACompiler::BuildSharedTensor (20 lines), variant dispatch arms in inferReturnTypes and compileBitcode (6 lines each)
- `python/src/clang_compiler.h` — ParseSharedBasis declaration, CUDACompiler::BuildSharedTensor declaration

## Decisions Made
- BuildSharedLinearLayout uses a lambda `buildBasisCarrier` for constructing OffsetBases/BlockBases NTTPs — avoids code duplication between the two carrier types while keeping the NTTP construction logic self-contained
- ParseSharedBasis is a separate method from ParseBasis because OffsetBases/BlockBases carriers are top-level templates (not nested in a LayoutFactory), so the APValue structure differs slightly — the outermost struct field layout is the same but the template parent context differs
- DispatchTypeParsing places the SharedTensorTemplateType check BEFORE TensorTemplateType to prevent SharedTensor types from accidentally matching the TensorTemplateType branch

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing ParseSharedBasis declaration in clang_compiler.h**
- **Found during:** Task 2 (ParseSharedBasis implementation)
- **Issue:** The LSP reported "out-of-line definition of ParseSharedBasis does not match any declaration in TypeInspector" — the header declared ParseSharedTensorType and DispatchTypeParsing but not the ParseSharedBasis helper
- **Fix:** Added `llvm::SmallVector<uint32_t, 4> ParseSharedBasis(const clang::TemplateArgument &Arg);` to TypeInspector in clang_compiler.h after ParseBasis
- **Files modified:** python/src/clang_compiler.h
- **Verification:** Build (`CC=clang CXX=clang++ bash build.sh`) succeeds
- **Committed in:** `3908028eec` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minor header addition — no scope creep.

## Issues Encountered
None.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- Plan 04-03 ready: pytest harness (test_shared_tensor.py) can now drive the full TypeBuilder → TypeInspector round-trip using the methods implemented in this plan
- The build/link/test verification chain is intact: 6/6 existing extern-call tests pass, and `llvm.SharedTensorParameter()` constructs successfully
- Plan 04-03's pytest harness can construct SharedTensorParameter objects, parse synthetic .cu sources, and verify round-trip correctness

---
*Phase: 04-c-templates-clang-ast-foundation*
*Completed: 2026-07-12*
