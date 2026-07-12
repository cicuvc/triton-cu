---
phase: 04-c-templates-clang-ast-foundation
plan: 01
subsystem: compiler
tags: [cuda, clang-ast, shared-memory, pybind11, device-templates, nttp]

requires:
  - phase: 02-semantic-time-inference
    provides: TypeBuilder, TypeInspector, FunctionResolver, TensorParameter, CudaFuncRequest::ParamTypes variant infrastructure
provides:
  - SharedLinearLayout and SharedTensor C++ device templates in tt_plugin.cu
  - SharedLayoutInfo / SharedTensorParameter C++ structs in clang_compiler.h
  - Extended TypeBuilder/TypeInspector with shared-tensor path declarations
  - llvm.SharedTensorParameter pybind11 binding
affects: [04-c-templates-clang-ast-foundation-plans-02-03, 05-ods-relaxation, 06-mlir-lowering, 07-e2e-gpu]

tech-stack:
  added: []
  patterns:
    - NTTP carrier structs (OffsetBases, BlockBases) using IntTuple<RANK> basis rows
    - auto NTTP parameters for SharedLinearLayout template flexibility
    - ShapeDims helper extracts variadic Shape<DIMS...> into constexpr array
    - Parallel structs mirroring existing TensorParameter/LayoutInfo pattern
    - pybind11 def_property pattern for nested struct member access

key-files:
  created: []
  modified:
    - python/test/gluon/tt_plugin.cu - Shared memory device templates
    - python/src/clang_compiler.h - SharedLayoutInfo, SharedTensorParameter, TypeBuilder/TypeInspector extensions
    - python/src/clang_compiler.cc - Updated out-of-line definitions for new variant types
    - python/src/llvm.cc - SharedTensorParameter pybind11 binding

key-decisions:
  - "OffsetBases uses RANK + N_BASES template params for NTTP structural type compatibility"
  - "SharedTensor uses void* sentinel member before T data[] for nvcc FAM compliance"
  - "DispatchTypeParsing return variant extended with SharedTensorParameter arm rather than parallel dispatch path"

patterns-established:
  - "ShapeDims<TShape>::All: extract variadic Shape<DIMS...> into constexpr array for stride computation"
  - "NTTP carrier struct pattern: OffsetBases/BlockBases mirror BasisGroup but as standalone top-level templates"
  - "SharedLinearLayout::evaluate() mirrors BasisGroup::evaluate() fold-expression pattern on NTTP Dims arrays"

requirements-completed:
  - SHTYPE-01
  - SHTYPE-02
  - SHAST-01

coverage:
  - id: D1
    description: "SharedLinearLayout C++ device template with evaluate() computing byte offsets from offset/block bases"
    requirement: SHTYPE-01
    verification:
      - kind: unit
        ref: "clang++ --cuda-device-only /tmp/test_shared_templates.cu"
        status: pass
      - kind: unit
        ref: "grep confirm SharedLinearLayout, evaluate() signature, Shape-independent"
        status: pass
    human_judgment: false
  - id: D2
    description: "SharedTensor<T, TShape, TLayout> C++ device template with variadic operator() → T& for read+write"
    requirement: SHTYPE-02
    verification:
      - kind: unit
        ref: "clang++ --cuda-device-only /tmp/test_shared_templates.cu"
        status: pass
      - kind: unit
        ref: "grep confirm T data[], operator()(auto... indices), T& return"
        status: pass
    human_judgment: false
  - id: D3
    description: "SharedLayoutInfo + SharedTensorParameter structs in clang_compiler.h, extended ParamTypes variant, TypeBuilder/TypeInspector declarations"
    requirement: SHAST-01
    verification:
      - kind: unit
        ref: "CC=clang CXX=clang++ bash build.sh — clang_compiler.cc compiles"
        status: pass
      - kind: unit
        ref: "grep confirm struct definitions, variant extension, method declarations"
        status: pass
    human_judgment: false
  - id: D4
    description: "llvm.SharedTensorParameter pybind11 binding with all five attributes (type, shape, offset_basis, block_basis, alignment) readable and writable"
    verification:
      - kind: unit
        ref: "python smoke test: construct, read/write all attributes, alignment set/get"
        status: pass
    human_judgment: false
  - id: D5
    description: "Test device functions write_shared_1d and process_shared_2d using SharedTensor& parameters"
    verification:
      - kind: unit
        ref: "clang++ --cuda-device-only /tmp/test_shared_templates.cu"
        status: pass
    human_judgment: false

duration: 8min
completed: 2026-07-12
status: complete
---

# Phase 04 Plan 01: C++ Templates + Clang AST Foundation Summary

**Device-side SharedLinearLayout/SharedTensor templates, clang AST bridge structs, and Python SharedTensorParameter binding — establishing the type system for shared-memory interop**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-12T14:22:48Z
- **Completed:** 2026-07-12T14:31:21Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- SharedLinearLayout and SharedTensor C++ device templates in tt_plugin.cu (OffsetBases, BlockBases, evaluate(), variadic operator() → T&)
- SharedLayoutInfo / SharedTensorParameter C++ structs in clang_compiler.h with extended ParamTypes variant
- TypeBuilder/TypeInspector forward declarations for shared-tensor path (BuildSharedLinearLayout, BuildSharedTensor, ParseSharedTensorType)
- llvm.SharedTensorParameter pybind11 binding with all five attributes readable/writable
- Test device functions (write_shared_1d, process_shared_2d) for Plan 04-03 pytest harness

## Task Commits

Each task was committed atomically:

1. **Task 1: Define OffsetBases, BlockBases, SharedLinearLayout, and SharedTensor in tt_plugin.cu** — `bd8dad01f0` (feat)
2. **Task 2: Declare SharedLayoutInfo, SharedTensorParameter, and TypeBuilder/TypeInspector extensions** — `4029f78ae1` (feat)
3. **Task 3: Expose llvm.SharedTensorParameter pybind11 binding** — `fcf18c0ee8` (feat)

## Files Modified
- `python/test/gluon/tt_plugin.cu` — Added OffsetBases, BlockBases, SharedLinearLayout (with evaluate()), SharedTensor (with variadic operator()→T&), ShapeDims helper, and test device functions
- `python/src/clang_compiler.h` — Added SharedLayoutInfo, SharedTensorParameter structs; extended CudaFuncRequest::ParamTypes variant; added SharedLinearLayoutTemplateType/SharedTensorTemplateType members + BuildSharedLinearLayout/BuildSharedTensor/ParseSharedTensorType method declarations; extended DispatchTypeParsing, EvaluateFunctionReturnType, TupleType variants
- `python/src/clang_compiler.cc` — Updated out-of-line definitions for DispatchTypeParsing, LookupFunctionWithPlaceholderFallback, EvaluateFunctionReturnType to match extended variant types
- `python/src/llvm.cc` — Added py::class_<SharedTensorParameter> binding with def_readwrite (type, shape) and def_property (offset_basis, block_basis, alignment)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] clang_compiler.cc out-of-line definitions mismatched after header variant extension**
- **Found during:** Task 2 (clang_compiler.h compilation)
- **Issue:** Changing `std::variant<ScalarType, TensorParameter>` to include `SharedTensorParameter` broke out-of-line definitions in clang_compiler.cc that used the old variant types. DispatchTypeParsing, LookupFunctionWithPlaceholderFallback, EvaluateFunctionReturnType, and TupleType::Types all used the 2-element variant.
- **Fix:** Updated all four out-of-line definitions and TupleType::Types to include SharedTensorParameter in the variant. The existing `std::get_if<TensorParameter>` call sites are unaffected — they correctly return nullptr for SharedTensorParameter values.
- **Files modified:** python/src/clang_compiler.cc, python/src/clang_compiler.h (TupleType)
- **Verification:** Full build (`CC=clang CXX=clang++ bash build.sh`) succeeds.
- **Committed in:** `4029f78ae1` (Task 2 commit)

**2. [Rule 3 - Blocking] nvcc rejected flexible array member in otherwise-empty SharedTensor struct**
- **Found during:** Task 1 (CUDA syntax check)
- **Issue:** nvcc requires at least one named member before a flexible array member (`T data[]`). The original design had SharedTensor with only `T data[]`, which nvcc rejects with "a flexible array member cannot be declared in an otherwise-empty type".
- **Fix:** Added `void* __shared_memory_base` sentinel member before `T data[]`. This struct is never allocated — it solely aliases external shared memory. The data[] member remains at a known offset for Phase 6 lowering.
- **Files modified:** python/test/gluon/tt_plugin.cu
- **Verification:** nvcc FAM error eliminated; clang CUDA compilation of minimal fragment succeeds; grep still finds `T data[]` in the file.
- **Committed in:** `bd8dad01f0` (Task 1 commit)

**3. [Design - Plan syntax adaptation] OffsetBases/BlockBases use RANK+N_BASES template params rather than RANK-only with class specializations**
- **Found during:** Task 1 (template design)
- **Issue:** The plan's illustrative syntax `OffsetBases<2>{IntTuple<2>{1,0}}` implies a single RANK template parameter with the array size set via class specialization. However, C++ NTTP requires structural types with fixed-size arrays, and partial specialization to set the array size for each combination of RANK and N_BASES is unwieldy for header-only device code where the number of bases varies per instantiation.
- **Fix:** Used `template<uint32_t RANK, uint32_t N_BASES>` for both OffsetBases and BlockBases, matching the existing `BasisGroup<N_BASES>` pattern. SharedLinearLayout accepts `auto OB, auto BB` NTTP parameters. The functional equivalent `OffsetBases<2, 1>{IntTuple<2>{1,0}}` compiles correctly.
- **Files modified:** python/test/gluon/tt_plugin.cu
- **Verification:** Clang CUDA compilation of minimal fragment with `SharedLinearLayout<OffsetBases<2,1>{IntTuple<2>{1,0}}, BlockBases<2,0>{}, 16>` succeeds.
- **Committed in:** `bd8dad01f0` (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 design adaptation)
**Impact on plan:** All three fixes are necessary for correctness and compilation. No scope creep. The OffsetBases design adapts the plan's illustrative syntax to C++20 NTTP requirements without changing semantics.

## Issues Encountered
- nvcc stricter than clang++ for C++20 features — pre-existing TArg/TRes errors (GNU extension brace-enclosed template args) are out of scope per deviation rules; our new templates compile correctly with clang CUDA mode which is the project's actual compiler.
- Default alignment from SharedLayoutInfo struct is 16 (not 0), matching the plan's D-01 specification. This is correct behavior.

## Next Phase Readiness
- Phase 04 Plan 02 ready: TypeBuilder implementations (BuildSharedLinearLayout, BuildSharedTensor) in clang_compiler.cc can use the header structs and declarations established here.
- Phase 04 Plan 03 ready: TypeInspector implementations (ParseSharedTensorType, DispatchTypeParsing shared branch) and pytest harness can use the Python binding and device templates.
- The SharedTensorParameter pybind binding is fully functional and can be consumed immediately by Plan 03's test harness.

---
*Phase: 04-c-templates-clang-ast-foundation*
*Completed: 2026-07-12*
