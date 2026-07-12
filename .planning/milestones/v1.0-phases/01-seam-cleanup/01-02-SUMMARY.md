---
phase: 01-seam-cleanup
plan: 02
subsystem: compiler
tags: [inference-hook, seam, coroutine, single-parse, parse-counter, Triton, CUDA, gl.call]
requires:
  - phase: 01-seam-cleanup
    plan: 01
    provides: "Dead code removed from compiler.py, _scalar_type_for helper replacing f64→Fp32 coercion"
provides:
  - "CUDA-backend inference-hook object (InferExternCallResult) exposed via codegen_fns[\"infer_extern_call_result\"]"
  - "Suspended CUDACompiler mechanism — parse once at semantic time, resume at llir stage"
  - "Per-compile parse-count delta assertion proving single-parse (SC3)"
  - "SuspendedCudaCompiler pybind11 class wrapping CUDACompiler with parse/compile_bitcode methods"
affects:
  - "Phase 02 (semantic-time inference consumes infer_result from hook)"
  - "Phase 03 (verification of single-parse + inference)"
tech-stack:
  added: []
  patterns:
    - "Inference hook as stateful object on backend via codegen_fns, not bare callable"
    - "Semantic→llir compiler travel via hook's _compilers dict on self._infer_hook"
    - "Shared LLVMContext across compile stages for correct bitcode linking"
    - "Per-compile parse counter delta spans hook creation through emit (not just llir stage)"
    - "pybind11 module_local context type requires lambda wrapper for method bindings"
key-files:
  created: []
  modified:
    - python/src/clang_compiler.h — compileBitcode declaration, getExternCudaParseCount declaration
    - python/src/clang_compiler.cc — static sExternCudaParseCount, increment in PerformParse, getExternCudaParseCount impl, CUDACompiler::compileBitcode method (3-phase: infer→codegen→emit)
    - python/src/llvm.cc — SuspendedCudaCompiler pybind11 class, get_extern_cuda_parse_count binding, parse lambda wrapper
    - third_party/nvidia/backend/compiler.py — InferExternCallResult class, hook in codegen_fns, suspended-compiler path in _pre_compile_extern_calls, LLVMContext sharing, parse assertion
    - python/triton/experimental/gluon/_runtime.py — .cu path pre-scan in make_ir with LLVM context creation
key-decisions:
  - "Parse counter snapshot taken at InferExternCallResult.__init__ so delta spans both semantic and llir stages"
  - "parse() binding uses lambda wrapper because member-function-pointer binding fails with py::module_local() llvm::LLVMContext"
  - "Metadata keys must not start with underscore (namedtuple field-name constraint in CompiledKernel)"
  - "Shared LLVMContext stored on InferExternCallResult._llvm_ctx and reused by make_llir + _pre_compile_extern_calls"
patterns-established:
  - "Inference hook object pattern: stateful object on backend instance, created in get_codegen_implementation, accessed by both semantic (_runtime.py) and llir (compiler.py) stages"
  - "Suspended compiler pattern: CUDACompiler created+parsed at semantic time, coroutine parked in HandleTranslationUnit, resumed at llir stage via compile_bitcode"
  - "Parse counter guard pattern: snapshot at compile start, delta computed at emit, asserted in make_llir against distinct .cu file count"
requirements-completed:
  - INFER-06
  - INFER-07
coverage:
  - id: D1
    description: "Inference hook (InferExternCallResult) exposed via codegen_fns[\"infer_extern_call_result\"] with create_and_suspend, infer_result, compile_bitcode (SC1)"
    requirement: INFER-06
    verification:
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py -x --tb=short"
        status: pass
    human_judgment: false
  - id: D2
    description: "Suspended CUDACompiler — parse once at semantic time, resume at llir stage via compile_bitcode (INFER-07, D-03)"
    requirement: INFER-07
    verification:
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py -x --tb=short"
        status: pass
    human_judgment: false
  - id: D3
    description: "Per-compile parse-count delta assertion in make_llir guards against double-parsing (SC3)"
    requirement: INFER-07
    verification:
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py -x --tb=short"
        status: pass
    human_judgment: false
  - id: D4
    description: "All 4 existing extern-call tests pass unchanged with new suspended-compiler path (SC5)"
    verification:
      - kind: integration
        ref: "pytest python/test/gluon/test_extern_call.py -x --tb=short"
        status: pass
    human_judgment: false
duration: 13 min
completed: 2026-07-11
status: complete
---

# Phase 01 Plan 02: Inference Hook & Single-Parse Seam Summary

**Inference-hook object (InferExternCallResult) exposed via CUDA backend codegen_fns; suspended CUDACompiler parses .cu once at semantic time and resumes at llir stage; parse-counter delta assertion proves single-parse; all 4 existing tests pass.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-07-11T10:35:50Z
- **Completed:** 2026-07-11T10:48:30Z
- **Tasks:** 3 (plus 4 auto-fix iterations)
- **Files modified:** 5

## Accomplishments
- C++ parse counter (sExternCudaParseCount) increments on each PerformParse; getExternCudaParseCount exposed via Python binding
- CUDACompiler::compileBitcode method splits inference+codegen+emit from monolithic tritonCompileCuda — works on already-parsed compiler
- SuspendedCudaCompiler pybind11 class wraps CUDACompiler with constructor, parse(), compile_bitcode()
- InferExternCallResult hook object created in get_codegen_implementation, stored on self._infer_hook, keyed as codegen_fns["infer_extern_call_result"]
- make_ir pre-scans kernel source for gl.call(".cu") patterns, creates+suspends CUDACompiler per distinct .cu
- _pre_compile_extern_calls prefers suspended-compiler path over old compile_cuda_to_module
- Shared LLVMContext across semantic→llir stages ensures correct bitcode linking
- Per-compile parse-count delta assertion in make_llir guards against double-parsing (SC3)
- All 4 existing extern-call tests pass unchanged (SC5)

## Task Commits

1. **Task 1: C++ parse counter + compileBitcode** — `1b06341ee8` (feat)
2. **Task 2: Python bindings — SuspendedCudaCompiler + parse counter** — `cdbcff6adb` (feat)
3. **Task 3: Inference hook + backend plumbing + parse assertion** — `199b8e01d3` (feat)
4. **Fix: LLVM context type mismatch (Rule 3)** — `74e3f3fa5e` (fix)
5. **Fix: parse counter snapshot + lambda wrapper (Rule 1/3)** — `4086cbf8fb` (fix)
6. **Fix: metadata keys for namedtuple compat (Rule 1)** — `fec1955d27` / `9abdfe386a` (fix)

## Files Created/Modified
- `python/src/clang_compiler.h` — compileBitcode declaration on CUDACompiler struct; getExternCudaParseCount in Public API section
- `python/src/clang_compiler.cc` — static sExternCudaParseCount counter; increment in PerformParse; getExternCudaParseCount impl; CUDACompiler::compileBitcode (3-phase method: inference→codegen→emit)
- `python/src/llvm.cc` — SuspendedCudaCompiler pybind11 class (constructor, parse lambda, compile_bitcode); get_extern_cuda_parse_count binding
- `third_party/nvidia/backend/compiler.py` — InferExternCallResult class (~60 lines); hook creation in get_codegen_implementation; suspended-compiler path in _pre_compile_extern_calls; shared LLVMContext reuse; parse assertion in make_llir
- `python/triton/experimental/gluon/_runtime.py` — .cu path pre-scan in make_ir; LLVM context creation for create_and_suspend

## Decisions Made
- Parse counter snapshot taken at InferExternCallResult.__init__ time (hook creation) so the per-compile delta spans both semantic-stage parses (create_and_suspend) and llir-stage compilations
- parse() binding uses a lambda wrapper instead of member-function-pointer because the latter is incompatible with pybind11's py::module_local() registration of llvm::LLVMContext
- Metadata keys use plain names (extern_parse_delta, extern_distinct_cu) — both double-underscore and single-underscore prefixed keys break namedtuple field-name validation in CompiledKernel.__init__
- Shared LLVMContext stored on InferExternCallResult._llvm_ctx from first create_and_suspend call, reused by make_llir and _pre_compile_extern_calls for correct bitcode linking

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] LLVM context type mismatch in make_ir**
- **Found during:** Task 3 verification (regression gate)
- **Issue:** make_ir receives MLIR ir.context, but PerformParse requires llvm::LLVMContext. The plan assumed make_ir's context parameter was an LLVM context.
- **Fix:** make_ir now explicitly imports triton._C.libtriton.llvm and creates llvm.context() for create_and_suspend. InferExternCallResult stores the shared LLVMContext (_llvm_ctx) so make_llir and _pre_compile_extern_calls reuse it.
- **Files modified:** python/triton/experimental/gluon/_runtime.py, third_party/nvidia/backend/compiler.py
- **Committed in:** 74e3f3fa5e

**2. [Rule 1 - Bug] pybind11 parse() binding rejected llvm.context argument**
- **Found during:** After Rule 3 fix, parse() called with correct LLVM context type still failed
- **Issue:** Member-function-pointer binding (`&CUDACompiler::PerformParse`) was incompatible with pybind11's py::module_local() registration of llvm::LLVMContext. pybind11's type resolver rejected the argument even though printed types matched.
- **Fix:** Wrapped parse() in a lambda that explicitly takes llvm::LLVMContext& and calls compiler.PerformParse(ctx, module_name).
- **Files modified:** python/src/llvm.cc
- **Committed in:** 4086cbf8fb

**3. [Rule 1 - Bug] Parse counter delta was 0 because snapshot taken at wrong time**
- **Found during:** After lambda fix, assertion fired: parse delta 0 != distinct .cu 1
- **Issue:** _parse_count_before was taken at start of _pre_compile_extern_calls (llir stage), but the parse already happened at semantic time (create_and_suspend). Delta from llir-stage snapshot to llir-stage after-count was 0.
- **Fix:** Moved _parse_count_before snapshot to InferExternCallResult.__init__ (hook creation, before any semantic work). Delta now spans hook creation through emit — correctly counts semantic-time parses.
- **Files modified:** third_party/nvidia/backend/compiler.py
- **Committed in:** 4086cbf8fb

**4. [Rule 1 - Bug] Metadata keys starting with underscore broke namedtuple**
- **Found during:** After parse counter fix, CompiledKernel.__init__ crashed with ValueError
- **Issue:** Metadata dict keys `__extern_cuda_parse_count__` and `__extern_call_specs__` started with underscore, which Python's namedtuple rejects for field names. Single-underscore prefix (`_extern_parse_delta`) also rejected.
- **Fix:** Renamed keys to `extern_parse_delta` and `extern_distinct_cu` (no underscore prefix). Distinct .cu count computed from by_libpath dict already available in _pre_compile_extern_calls.
- **Files modified:** third_party/nvidia/backend/compiler.py
- **Committed in:** fec1955d27, 9abdfe386a

---

**Total deviations:** 4 auto-fixed (3 Rule 1 bug fixes, 1 Rule 3 blocking fix)
**Impact on plan:** All fixes necessary for correctness (type compatibility, timing, library constraints). No scope creep — the core design (hook, suspended compiler, parse assertion) remains exactly as planned.

## Issues Encountered
- pybind11 module_local type + member-function-pointer incompatibility on llvm::LLVMContext — worked around with lambda wrapper
- Python namedtuple rejects underscore-prefixed field names — metadata keys adjusted
- plan assumed MLIR context could substitute for LLVM context in make_ir — corrected with explicit llvm.context() creation

## Known Stubs
- `InferExternCallResult.infer_result()` raises `NotImplementedError` (intentional — result consumed in Phase 2). File: `third_party/nvidia/backend/compiler.py`

## Next Phase Readiness
- Inference hook seam established — Phase 02 can consume `infer_result` for return-type inference
- Single-parse guarantee is proven by the parse-counter delta assertion
- All 4 existing extern-call tests pass unchanged
- Ready for `02-PLAN.md` (Semantic-Time Inference)

---

*Phase: 01-seam-cleanup*
*Completed: 2026-07-11*
