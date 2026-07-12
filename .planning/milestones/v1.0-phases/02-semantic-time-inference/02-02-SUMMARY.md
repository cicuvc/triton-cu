---
phase: 02-semantic-time-inference
plan: 02
subsystem: compiler
tags: [cuda, clang, pybind11, return-type-inference, python-binding]

requires:
  - phase: 02-semantic-time-inference
    plan: 01
    provides: CUDACompiler::inferReturnTypes C++ method, PlaceholderLayout, BuildTensor placeholder branch

provides:
  - SuspendedCudaCompiler.infer() pybind11 binding calling CUDACompiler::inferReturnTypes
  - InferExternCallResult.infer_result() filled implementation replacing NotImplementedError stub
  - Per-result (scalar_name, shape) tuple extraction from C++ TensorParameter data

affects:
  - 02-03 (semantic-time consumption via call_extern)

tech-stack:
  added: []
  patterns:
    - pybind11 binding pattern for inference-only method mirroring compile_bitcode structure
    - PlaceholderLayout-probed CudaFuncRequest construction with empty bases + n_warps=0
    - ScalarType↔name dual mapping pattern (dtype→ScalarType for request, ScalarType→name for result)

key-files:
  created: []
  modified:
    - python/src/llvm.cc - SuspendedCudaCompiler.infer() pybind11 binding after compile_bitcode
    - third_party/nvidia/backend/compiler.py - InferExternCallResult.infer_result() filled from stub

key-decisions:
  - "Inlined _scalar_type_for in infer_result rather than extracting from _pre_compile_extern_calls to minimize blast radius"

requirements-completed: [INFER-03]

coverage:
  - id: D1
    description: "SuspendedCudaCompiler.infer() pybind11 binding in llvm.cc"
    requirement: INFER-03
    verification:
      - kind: unit
        ref: "grep -c '\.def(\"infer\"' python/src/llvm.cc → 1"
        status: pass
      - kind: unit
        ref: "grep -c 'compiler.inferReturnTypes(requests)' python/src/llvm.cc → 1"
        status: pass
    human_judgment: false
  - id: D2
    description: "InferExternCallResult.infer_result() filled implementation in compiler.py"
    requirement: INFER-03
    verification:
      - kind: unit
        ref: "grep -c 'def infer_result.*libpath' third_party/nvidia/backend/compiler.py → 1"
        status: pass
      - kind: unit
        ref: "grep -c 'compiler.infer(' third_party/nvidia/backend/compiler.py → 1"
        status: pass
      - kind: unit
        ref: "grep -c 'raise NotImplementedError' third_party/nvidia/backend/compiler.py → 2 (only f64 guards remain)"
        status: pass
      - kind: unit
        ref: "grep -c '_scalar_names' third_party/nvidia/backend/compiler.py → 2"
        status: pass
      - kind: unit
        ref: "grep -c 'n_warps = 0' third_party/nvidia/backend/compiler.py → 1"
        status: pass
    human_judgment: false

duration: 0min
completed: 2026-07-11
status: complete
---

# Phase 02 Plan 02: Python Binding & Hook for Semantic-Time Inference

**Pybind11 `SuspendedCudaCompiler.infer()` binding + `InferExternCallResult.infer_result()` filled hook calling CUDACompiler::inferReturnTypes**

## Performance

- **Duration:** < 1 min
- **Started:** 2026-07-11T12:15:19Z
- **Completed:** 2026-07-11T12:16:18Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `SuspendedCudaCompiler.infer()` pybind11 method binding in `python/src/llvm.cc` — wraps `CUDACompiler::inferReturnTypes`, returns `(ok, bitcode, error, results)` with empty bitcode/mangled names (pure type inspection, no codegen)
- Filled `InferExternCallResult.infer_result(libpath, func, arg_params, use_fast_math)` — builds `CudaFuncRequest` with PlaceholderLayout-probed `TensorParameter` args (empty bases + `n_warps=0`), calls the new `infer()` binding, extracts per-result `(scalar_name, shape)` tuples
- ScalarType↔name mapping correct for all 5 supported types: Fp32→f32, Fp16→f16, Bf16→bf16, Int32→i32, Int64→i64
- Inlined `_scalar_type_for` helper in `infer_result` mirrors the same logic from `_pre_compile_extern_calls`, reusing identical dtype→ScalarType mapping
- Error handling: `RuntimeError` on missing compiler, C++ inference failure, or zero results

## Task Commits

Each task was committed atomically:

1. **Task 1: Add SuspendedCudaCompiler.infer() pybind11 binding** - `6fd97286ec` (feat)
2. **Task 2: Fill InferExternCallResult.infer_result() hook** - `ededad87b3` (feat)

## Files Modified

- `python/src/llvm.cc` — Added `.def("infer", ...)` to `SuspendedCudaCompiler` pybind11 class (23 lines after `compile_bitcode` at line 1066). Calls `compiler.inferReturnTypes(requests)`, returns 4-tuple convention with empty `MangledName`/`ExtractorMangledNames`.
- `third_party/nvidia/backend/compiler.py` — Replaced `infer_result` stub (5 lines, `raise NotImplementedError`) with full implementation (89 lines). New signature includes `libpath` parameter for compiler dictionary lookup. PlaceholderLayout args constructed with empty bases + `n_warps=0`.

## Decisions Made

- Inlined `_scalar_type_for` helper in `infer_result` rather than extracting the existing local function from `_pre_compile_extern_calls` — minimizes blast radius; the identical dtype→ScalarType mapping is maintained in both locations

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Build integration deferred to Plan 02-03 (full build + test after all plans complete)
- Ready for Plan 02-03: semantic-time consumption via `call_extern` — the `infer_result` hook and `SuspendedCudaCompiler.infer()` binding are now available

## Known Stubs

None.

---

*Phase: 02-semantic-time-inference*
*Completed: 2026-07-11*
