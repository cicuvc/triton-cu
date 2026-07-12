---
phase: 01-seam-cleanup
verified: 2026-07-11T18:00:00Z
status: passed
score: 11/11 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 01: Seam & Cleanup Verification Report

**Phase Goal:** Establish a clean way for the backend-agnostic Gluon semantic layer to invoke CUDA return-type inference, ensure the `.cu` is not parsed twice, and clear the bundled bugs — before touching the inference data flow.
**Verified:** 2026-07-11
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | CUDA backend exposes return-type-inference callable via `get_codegen_implementation` / `codegen_fns` | ✓ VERIFIED | `InferExternCallResult` class (compiler.py:190–252) with `infer_result(func, arg_params, use_fast_math)` at line 220 (raises `NotImplementedError` in Phase 1 per prohibition). Exposed as `codegen_fns["infer_extern_call_result"]` at compiler.py:342. |
| SC2 | Non-CUDA/interpreter paths degrade gracefully (no crash when hook absent) | ✓ VERIFIED | `_runtime.py:91`: `codegen_fns.get("infer_extern_call_result")` returns `None` if key absent. `_runtime.py:92`: `if _hook is not None:` guards. `compiler.py:681`: `getattr(self, '_infer_hook', None)` with default. Non-CUDA backends skip the pre-scan entirely. |
| SC3 | Single clang parse per `.cu` (reuse/cache, verified by parse counter delta) | ✓ VERIFIED | `sExternCudaParseCount` incremented in `PerformParse` (clang_compiler.cc:692). `InferExternCallResult.__init__` snapshots base count (compiler.py:201). `_pre_compile_extern_calls` computes per-compile delta (compiler.py:773–776). `make_llir` asserts `parse_count_delta == distinct_cu` (compiler.py:586–592). Suspended CUDACompiler: created+parsed at semantic time (`_runtime.py:99`), resumed at llir stage (`compiler.py:705`) — no second parse. |
| SC4 | Dead code at compiler.py:510–513 removed; f64/fp64 raises clear error | ✓ VERIFIED | Only 1 `del llvm_mod` at line 611 (legitimate). `grep -c 'f64.*Fp32\|fp64.*Fp32'` returns 0. `_semantic.py:260–263`: early dtype-string guard raises `NotImplementedError`. `compiler.py:657–664`: `_scalar_type_for` backstop raises `NotImplementedError` for f64/fp64/float64. |
| SC5 | All 4 existing extern-call tests pass unchanged | ✓ VERIFIED | `pytest python/test/gluon/test_extern_call.py` — 4 passed in 1.77s. |

**Score:** 5/5 Success Criteria verified

### Plan 01-01 Must-Haves

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | f64/fp64/float64 args raise NotImplementedError before IR build | ✓ VERIFIED | `_semantic.py:260–263`: dtype-string check `("fp64", "f64", "float64")` before `first_input` assignment (line 265). No CUDA import in guard area. |
| 2 | `dtype_to_scalar` no longer maps f64/fp64 to Fp32 — `_scalar_type_for` raises | ✓ VERIFIED | `compiler.py:657–664`: `_scalar_type_for` looks up in `dtype_to_scalar` (no f64 entries at lines 648–654), raises `NotImplementedError` for f64/fp64/float64. `grep -c 'f64.*Fp32\|fp64.*Fp32'` → 0. |
| 3 | Unreachable duplicate return block at compiler.py:510–513 does not exist | ✓ VERIFIED | Only 1 `del llvm_mod` at line 611. Single `return ret` at line 613. `_pre_compile_extern_calls` now starts at line 615 (shifted up 4 lines from original 515). |
| 4 | All 4 existing extern-call tests pass unchanged | ✓ VERIFIED | `pytest` — 4 passed. |

**Score:** 4/4 Plan 01-01 truths verified

### Plan 01-02 Must-Haves

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `get_codegen_implementation` returns dict with `infer_extern_call_result` key | ✓ VERIFIED | `compiler.py:342`: `codegen_fns["infer_extern_call_result"] = _infer_hook`. `InferExternCallResult` instance stored on `self._infer_hook` (line 341). |
| 2 | Parse counter increments on `ExecuteAction`; `get_extern_cuda_parse_count()` accessible from Python | ✓ VERIFIED | C++: `sExternCudaParseCount` at clang_compiler.cc:57, incremented in `PerformParse` at line 692, `getExternCudaParseCount()` at line 702. Python: `llvm.cc:1021–1023` binds `m.def("get_extern_cuda_parse_count", ...)`. |
| 3 | `make_ir` pre-scans `.cu` paths, creates SuspendedCudaCompiler, calls `parse()` | ✓ VERIFIED | `_runtime.py:70–99`: regex scans `gl.call(".cu")` patterns, resolves absolute paths, calls `_hook.create_and_suspend(_cu_source, _llvm_ctx, _cu_path)`. |
| 4 | Coroutine lifetime per-compile (D-04): created+suspended at semantic, consumed at llir, torn down | ✓ VERIFIED | Created in `_runtime.py:99` (semantic time). Consumed in `compiler.py:705` (llir stage). No cross-kernel reuse — `_compilers` dict keyed by libpath, `create_and_suspend` is idempotent per compile. Disk cache via `.cu` path in cache key handles repeated compiles. |
| 5 | `make_llir` asserts per-compile parse-count delta equals distinct `.cu` count | ✓ VERIFIED | `compiler.py:586–592`: `assert parse_count_delta == distinct_cu`. Delta computed at `compiler.py:773–776` from `InferExternCallResult.__init__` snapshot (line 201). Assertion survives pytest multi-test (per-compile delta, not live global counter). |
| 6 | All 4 existing extern-call tests pass unchanged after C++ rebuild | ✓ VERIFIED | `pytest` — 4 passed. |
| 7 | `infer_result` raises `NotImplementedError` (callable but result not consumed in Phase 1) | ✓ VERIFIED | `compiler.py:225–226`: `raise NotImplementedError("infer_result: return-type inference not available in Phase 1")`. |

**Score:** 7/7 Plan 01-02 truths verified

### Prohibitions

| # | Prohibition | Status | Evidence |
|---|-------------|--------|----------|
| P1 | `call_extern` must NOT consume the inference hook to build result types in Phase 1 (still infers from `first_input`) | ✓ VERIFIED | `_semantic.py:265–270`: infers result type from `first_input.dtype` and `first_input.shape`. No reference to `infer_result` or `infer_extern_call_result` in `_semantic.py`. |
| P2 | No silent f64/fp64 → Fp32 coercion remains anywhere | ✓ VERIFIED | `grep -c 'f64.*Fp32\|fp64.*Fp32' compiler.py` → 0. Both layers raise `NotImplementedError`. |
| P3 | No new tests added in Phase 1 (tests are Phase 3) | ✓ VERIFIED | `test_extern_call.py` still has 4 tests (same count as before). |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `third_party/nvidia/backend/compiler.py` | Dead code removed, `_scalar_type_for` helper, `InferExternCallResult` class, hook exposure, suspended-compiler path, parse assertion | ✓ VERIFIED | Lines 190–252: `InferExternCallResult`. Lines 316–343: hook in `get_codegen_implementation`. Lines 648–664: `dtype_to_scalar` + `_scalar_type_for`. Lines 677–778: suspended-compiler path + delta. Lines 582–592: parse assertion. |
| `python/triton/experimental/gluon/language/_semantic.py` | f64 early guard before `first_input` | ✓ VERIFIED | Lines 259–263: dtype-string guard. Still infers from `first_input` at lines 265–270. |
| `python/src/clang_compiler.h` | `compileBitcode` declaration, `getExternCudaParseCount` | ✓ VERIFIED | Lines 346–349: `compileBitcode` method. Line 462: `getExternCudaParseCount()` in Public API. |
| `python/src/clang_compiler.cc` | Static parse counter, increment in `PerformParse`, `getExternCudaParseCount`, `compileBitcode` (3-phase) | ✓ VERIFIED | Line 57: `sExternCudaParseCount`. Line 692: increment. Line 702: `getExternCudaParseCount`. Lines 810–935: `compileBitcode` (Phase 1 inference: 831–889, Phase 2 codegen: 892–922, Phase 3 emit: 924–935). |
| `python/src/llvm.cc` | `SuspendedCudaCompiler` pybind11 class, `get_extern_cuda_parse_count` binding | ✓ VERIFIED | Lines 1021–1023: `get_extern_cuda_parse_count`. Lines 1029–1066: `SuspendedCudaCompiler` with constructor, `parse` lambda, `compile_bitcode` lambda. |
| `python/triton/experimental/gluon/_runtime.py` | `.cu` pre-scan in `make_ir`, suspended compiler creation | ✓ VERIFIED | Lines 67–99: pre-scan regex, path resolution, `create_and_suspend` guarded by `codegen_fns.get("infer_extern_call_result")`. |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `make_ir` → `InferExternCallResult` | `_runtime.py:91` → `compiler.py:342` | `codegen_fns.get("infer_extern_call_result")` → `_hook` | ✓ WIRED |
| `create_and_suspend` → `SuspendedCudaCompiler.parse()` | `compiler.py:203–218` → `llvm.cc:1035–1040` | `infer_hook.create_and_suspend(source, llvm_context, libpath)` builds compiler, calls `parse()` | ✓ WIRED |
| `_pre_compile_extern_calls` → `InferExternCallResult.compile_bitcode` | `compiler.py:681–711` → `compiler.py:228–252` | `self._infer_hook.compile_bitcode(libpath, requests)` | ✓ WIRED |
| `make_llir` → parse delta assertion | `compiler.py:582–592` → `compiler.py:773–776` | `metadata["extern_parse_delta"] == metadata["extern_distinct_cu"]` | ✓ WIRED |
| `call_extern` → f64 guard | `_semantic.py:259–263` | `NotImplementedError` before `first_input` assignment | ✓ WIRED |
| `dtype_to_scalar` → `_scalar_type_for` | `compiler.py:648–664` | `_scalar_type_for` wraps `dtype_to_scalar.get()`, raises for f64 | ✓ WIRED |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `InferExternCallResult._parse_count_before` | `self._parse_count_before` | `llvm.get_extern_cuda_parse_count()` (C++ counter) | Yes — real global counter | ✓ FLOWING |
| Suspended compiler `_compilers` dict | `self._compilers[libpath]` | `create_and_suspend` → `llvm.SuspendedCudaCompiler` | Yes — real compiler with parsed AST | ✓ FLOWING |
| `compile_bitcode` result | `bitcode, mangled_names, extractor_names, ret_types_list` | `SuspendedCudaCompiler.compile_bitcode(requests)` → `EmitFinalModule` | Yes — real LLVM bitcode | ✓ FLOWING |
| `metadata["extern_parse_delta"]` | `_parse_delta` | `get_extern_cuda_parse_count() - _count_before` | Yes — per-compile delta | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Existing extern-call tests pass | `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" pytest python/test/gluon/test_extern_call.py -x --tb=short` | 4 passed in 1.77s | ✓ PASS |
| No f64→Fp32 coercion strings in compiler.py | `grep -c 'f64.*Fp32\|fp64.*Fp32' third_party/nvidia/backend/compiler.py` | 0 | ✓ PASS |
| Dead code removed — only 1 `del llvm_mod` | `grep -c 'del llvm_mod' third_party/nvidia/backend/compiler.py` | 1 | ✓ PASS |
| f64 guard present in _semantic.py | `grep -c 'float64' python/triton/experimental/gluon/language/_semantic.py` | 2 | ✓ PASS |

### Probe Execution

No probes declared for this phase — skipped.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| INFER-06 | 01-02 | Gluon semantic layer reaches CUDA-specific inference through codegen_fns hook, without importing NVIDIA backend code. Interpreter/non-CUDA backends degrade gracefully. | ✓ SATISFIED | `codegen_fns["infer_extern_call_result"]` at compiler.py:342. `_runtime.py:91` uses `codegen_fns.get()` (graceful). No NVIDIA imports in `_runtime.py` or `_semantic.py`. |
| INFER-07 | 01-02 | No redundant clang parse — inference at semantic time and bitcode compilation at llir stage share single parse. | ✓ SATISFIED | Per-compile parse delta assertion (compiler.py:582–592). Suspended CUDACompiler parses once at semantic time, resumed at llir stage. `get_extern_cuda_parse_count()` binding + C++ counter. |
| BUG-01 | 01-01 | Remove dead unreachable code at compiler.py:510–513. | ✓ SATISFIED | Only 1 `del llvm_mod` at line 611. Single `return ret` at line 613. Original dead block no longer exists. |
| BUG-02 | 01-01 | f64/fp64 handling: raise clear error. | ✓ SATISFIED | `_semantic.py:260–263`: early dtype-string guard raises `NotImplementedError`. `compiler.py:660–662`: `_scalar_type_for` backstop raises `NotImplementedError`. No silent coercion anywhere. |

### Orphaned Requirements

No orphaned requirements. All 4 requirement IDs (INFER-06, INFER-07, BUG-01, BUG-02) are covered by plans in this phase and REQUIREMENTS.md confirms they are mapped.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `compiler.py` | 104 | `# TODO: Handle non-"a" sms` | ℹ️ Info | Pre-existing — not introduced by this phase |
| `compiler.py` | 383 | `# TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass` | ℹ️ Info | Pre-existing — not introduced by this phase |

No new debt markers (TBD/FIXME/XXX) introduced in any phase-modified file. No stubs, no placeholder implementations, no empty returns in phase-specific code.

### Gaps Summary

No gaps found. All must-haves verified, all requirements satisfied, all prohibitions intact.

---

_Verified: 2026-07-11_
_Verifier: the agent (gsd-verifier)_
