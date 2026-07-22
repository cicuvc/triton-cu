---
phase: 07-e2e-verification
verified: 2026-07-22T00:00:00Z
status: human_needed
score: 1/9 must-haves verified
behavior_unverified: 8
overrides_applied: 0
overrides: []
behavior_unverified_items:
  - truth: "CUDA device functions in tt_plugin.cu instantiate successfully from Gluon kernels via gl.call() — Phase 7 E2E tests can invoke shared_accumulate and write_swizzled_2d without template substitution or void-return failures"
    test: "Run test_shared_read_write or test_shared_accumulate with a working libtriton.so build on the RTX 5090"
    expected: "gl.call() JIT compiles the device function without template substitution errors; kernel launches and returns expected GPU output"
    why_human: "GPU JIT compilation blocked by pre-existing LLVM dynamic-linking build issue (libLLVM.so.23.0git CLI option double-registration). Code structure verified via TDD file-content tests and plan-spec cross-reference — no syntax errors, template parameters match, operations match plan exactly."
  - truth: "write_swizzled_2d populates all 32×16 shared-memory elements with deterministic identity values — subsequent shared_memory_descriptor.load() after gl.barrier() recovers the per-element data for swizzle round-trip verification"
    test: "Run test_swizzle_round_trip with all 4 parametrized patterns on the RTX 5090"
    expected: "shm.load(identity_layout) recovers float(i*16+j) at each logical (i,j) after write_swizzled_2d writes i*16+j through the swizzled layout; Python evaluate_shared() reference matches bit-for-bit"
    why_human: "GPU execution blocked by LLVM build issue. Loop structure (i=0..31, j=0..15) and write operation (static_cast<T>(i*16+j)) verified via file-content analysis."
  - truth: "A Gluon kernel allocates shared memory, passes the descriptor to process_shared_2d via gl.call(), synchronizes with gl.barrier(), loads back via shm.load(), and stores to output — torch.testing.assert_close confirms write-back visibility"
    test: "Run test_shared_read_write on the RTX 5090"
    expected: "out == x * 2.0 (input seeded via shm.store(x), scaled by process_shared_2d with SCALE=2.0, barrier, shm.load(DIST_LAYOUT))"
    why_human: "GPU execution blocked by LLVM build issue. Kernel code structure verified: shm.store(x) → gl.call(process_shared_2d) → gl.barrier() → shm.load(dist_layout) → gl.store. All operations present and correctly ordered."
  - truth: "A Gluon kernel passes both a shared_memory_descriptor AND a distributed tensor to shared_accumulate in a single gl.call() — shared+distributed args flow through the mixed-arg lowering path correctly"
    test: "Run test_shared_accumulate on the RTX 5090"
    expected: "out == x (shared memory starts zero, shared_accumulate does shm(i) += val.data[i], barrier, shm.load(layout))"
    why_human: "GPU execution blocked by LLVM build issue. Kernel code structure verified: gl.allocate_shared_memory → gl.call(shared_accumulate, shm, vals) with both shared+distributed args → gl.barrier() → shm.load(). Mixed-arg path structurally present."
  - truth: "Byte-offset values computed via Python evaluate_shared() (replicating SharedLinearLayout::evaluate XOR logic) match kernel output bit-for-bit — for all 4 swizzle patterns at all 32×16 indices"
    test: "Run test_swizzle_round_trip with all 4 parametrized patterns on the RTX 5090; verify assert_close(out, compute_swizzle_expected(...)) for each"
    expected: "torch.testing.assert_close passes for all 4 patterns (identity, offset_only, cross_dim, full_xor)"
    why_human: "GPU execution blocked by LLVM build issue. Python evaluate_shared() faithfully replicates C++ SharedLinearLayout::evaluate() XOR-addition logic (tt_plugin.cu:160-166). All 4 parametrized patterns have correct offset_bases from RESEARCH.md. compute_swizzle_expected() correctly maps flat_index → swizzled_position → float(flat_index)."
  - truth: "All 6 existing test_extern_call.py tests pass unchanged after new test functions are added — no code path regression in the tensor-only lowering"
    test: "Run pytest python/test/gluon/test_extern_call.py -k 'not (shared_read_write or shared_accumulate or swizzle_round_trip)' on the RTX 5090"
    expected: "All 6 existing tests (elementwise_add, intra_warp_add_sibling, reduce_different_shape, split_add_tuple, reduce_f16_f32, gl_call_no_inference_hook_raises) pass"
    why_human: "GPU execution blocked by LLVM build issue. Existing test code (lines 1-167 of test_extern_call.py) is unmodified from pre-Phase-7 state. New code appended after line 169 with clear separator. No import changes, no shared-state modifications, no monkey-patching that could affect existing tests."
  - truth: "The Gluon lit suite (5 original + 1 Phase 6 extern-call-shared-args = 6 tests) passes unchanged — no MLIR/dialect-level regression"
    test: "Run cd build && ninja triton-opt && lit -v test/Gluon/ test/TritonGPU/extern-call-shared-args.mlir"
    expected: "All 6 lit tests pass (5 Gluon + 1 extern-call-shared-args)"
    why_human: "GPU execution blocked by LLVM build issue. Phase 7 test code is Python-level only — no MLIR/dialect-level changes were made. Lit tests are unaffected by this phase's file-only additions."
  - truth: "PTX for every shared-memory gl.call() contains ld.shared or st.shared — D-31 L-01 landmine automated assertion catches addrspace(3) pointer erasure"
    test: "Run test_shared_read_write, test_shared_accumulate, and test_swizzle_round_trip on the RTX 5090"
    expected: "D-31 assertion passes for all 3 tests — compiled.asm['ptx'] contains 'ld.shared' or 'st.shared'"
    why_human: "GPU execution blocked by LLVM build issue. D-31 PTX assertion code is structurally correct in all 3 test functions (test_extern_call.py:233-238, 253-258, 339-342), following the established test_lowerings.py:197-201 pattern."
human_verification:
  - test: "Run full GPU regression: PYTHONPATH=... python -m pytest python/test/gluon/test_extern_call.py python/test/gluon/test_shared_tensor.py -x -s --tb=short -n 8"
    expected: "All tests pass — 6 existing + 2 new SHTEST-01 + 4 parametrized SHTEST-02 + 4 Phase 4 = 16 total test passes"
    why_human: "Requires a working libtriton.so build on the RTX 5090; currently blocked by pre-existing LLVM dynamic-linking issue (libLLVM.so.23.0git CLI option double-registration with installed triton's static LLVM). Code structure and correctness verified via static analysis."
  - test: "Run Gluon lit suite: cd build && ninja triton-opt && lit -v test/Gluon/ test/TritonGPU/extern-call-shared-args.mlir"
    expected: "All 6 lit tests pass — zero MLIR/dialect-level regression"
    why_human: "Requires a working triton-opt build; blocked by same LLVM build environment issue. No Phase-7 changes to MLIR/dialect code — lit regression risk is minimal but must be confirmed before v1.1 ship."
  - test: "Verify D-31 L-01 landmine triggers on a fake failure: temporarily rename gl.barrier() to gl.barrier_missing() in one kernel, confirm PTX assertion fails"
    expected: "PTX assertion correctly fails when shared-memory instruction is absent"
    why_human: "D-31 assertion code is structurally correct but exercise of the landmine's failure path requires GPU execution; static analysis confirms the assertion pattern matches test_lowerings.py:197-201 exactly."
---

# Phase 7: E2E Verification Report

**Phase Goal:** Full pipeline works end-to-end — shared memory read+write through `gl.call()` produces correct GPU results, swizzle layouts round-trip correctly, and all existing tests pass without regression

**Verified:** 2026-07-22
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | CUDA device functions in tt_plugin.cu instantiate successfully from Gluon kernels via gl.call() | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | Code exists with correct signatures at tt_plugin.cu:227-242. TDD file-content tests (test_tt_plugin_functions.py) confirm template parameters, operations, placement. GPU JIT compilation blocked by LLVM build issue. |
| 2 | write_swizzled_2d populates all 32×16 shared-memory elements with deterministic identity values | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | Loop writes `static_cast<T>(i*16+j)` for i=0..31, j=0..15 at tt_plugin.cu:238-242. Operation matches plan spec exactly. GPU verification blocked. |
| 3 | shared_accumulate loop bound uses TLayout::REG_SIZE (not N from shape) — guarantees correct iteration for non-identity layouts | ✓ VERIFIED | TDD test `test_threat_mitigations` confirms `TLayout::REG_SIZE` at loop bound (tt_plugin.cu:233). Plan spec requirement satisfied. |
| 4 | Gluon kernel allocates shared memory, passes descriptor to process_shared_2d via gl.call(), synchronizes with gl.barrier(), loads back via shm.load(), stores to output — assert_close confirms write-back visibility | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | shared_read_write_kernel (test_extern_call.py:176-199): shm.store(x) → gl.call(process_shared_2d) → gl.barrier() → shm.load(dist_layout). Pattern correct. GPU verification blocked. |
| 5 | Gluon kernel passes both a shared_memory_descriptor AND a distributed tensor to shared_accumulate in a single gl.call() — mixed-arg path correct | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | shared_accumulate_kernel (test_extern_call.py:202-219): gl.call(shared_accumulate, shm, vals) with both shared (addrspace 3) and distributed (addrspace 0) args. GPU verification blocked. |
| 6 | Python evaluate_shared() (replicating SharedLinearLayout::evaluate XOR logic) matches kernel output bit-for-bit — for all 4 swizzle patterns at all 32×16 indices | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | Python evaluate_shared() (test_extern_call.py:280-290) faithfully replicates C++ XOR-addition (tt_plugin.cu:160-166). 4 parametrized patterns (identity, offset_only, cross_dim, full_xor) present. GPU verification blocked. |
| 7 | All 6 existing test_extern_call.py tests pass unchanged — no code path regression | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | Existing test code (lines 1-167) unmodified. New code appended after line 169 with clear separator. GPU regression run blocked. |
| 8 | Gluon lit suite (5 original + 1 Phase 6 = 6 tests) passes unchanged — no MLIR/dialect-level regression | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | No MLIR/dialect changes made in Phase 7. Lit run blocked by LLVM build issue. |
| 9 | PTX for every shared-memory gl.call() contains ld.shared or st.shared — D-31 L-01 landmine catches addrspace(3) pointer erasure | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | D-31 assertion code present in all 3 test functions (test_extern_call.py:233-238, 253-258, 339-342) matching test_lowerings.py:197-201 pattern. GPU PTX generation blocked. |

**Score:** 1/9 truths verified (8 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `python/test/gluon/tt_plugin.cu` | shared_accumulate and write_swizzled_2d device functions | ✓ VERIFIED | Lines 227-242: shared_accumulate with 4 template params + TLayout::REG_SIZE loop; write_swizzled_2d with nested 32×16 loop + static_cast<T>. Placed between process_shared_2d and END OF DEFINITIONS. Existing functions unmodified. |
| `python/test/gluon/test_extern_call.py` | Two SHTEST-01 E2E tests + one SHTEST-02 parametrized swizzle test + three @gluon.jit kernel functions | ✓ VERIFIED | Lines 170-343: 3 kernel functions, 2 Python helpers, 3 test functions (2 SHTEST-01 + 1 SHTEST-02 with 4 parametrized patterns). D-31 PTX assertions in all tests. All 6 original test functions (lines 1-167) unmodified. |
| `python/test/gluon/test_extern_call.py` | D-31 L-01 PTX landmine guard (ld.shared/st.shared assertions) | ✓ VERIFIED | Assertions at lines 233-238, 253-258, 339-342. Pattern matches test_lowerings.py:197-201 exactly. |
| `python/test/gluon/test_tt_plugin_functions.py` | TDD test harness for 07-01 | ✓ VERIFIED | 5 assertions PASS: function existence, signature verification, placement check, existing-function preservation, threat mitigation (T-07-01 REG_SIZE loop bound). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| test_shared_read_write → shared_read_write_kernel | gl.call() → tt_plugin.cu:process_shared_2d | result_layout=[] void-returning extern call + shm.store() → gl.call() → gl.barrier() → shm.load() chain | ✓ WIRED | test_extern_call.py:193-195 — gl.call("process_shared_2d", shm, SCALE, result_layout=[]); barrier at line 195. |
| test_shared_accumulate → shared_accumulate_kernel | gl.call() → tt_plugin.cu:shared_accumulate | Mixed-arg: shared_memory_descriptor + distributed tensor | ✓ WIRED | test_extern_call.py:214 — gl.call("shared_accumulate", shm, vals, result_layout=[]) passes both shared and distributed args. |
| test_swizzle_round_trip → swizzle_kernel | gl.call() → tt_plugin.cu:write_swizzled_2d | Writes i*16+j identity values; readback via shm.load(identity_layout) after barrier | ✓ WIRED | test_extern_call.py:269-271 — gl.call("write_swizzled_2d", shm, result_layout=[]); barrier at line 271. |
| test_swizzle_round_trip → evaluate_shared() Python reference | tt_plugin.cu:SharedLinearLayout::evaluate() | XOR-addition of basis rows for set bits | ✓ WIRED | Python evaluate_shared() (test_extern_call.py:280-290) faithfully replicates C++ evaluate() (tt_plugin.cu:160-166). compute_swizzle_expected() (lines 293-304) correctly maps flat_index → swizzled_position → float(flat_index). |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| shared_read_write_kernel | `x` (loaded input) | `gl.load(x_ptr + offs)` | Yes — loads from GPU global memory | ✓ FLOWING |
| shared_read_write_kernel | `shm` (shared memory) | `gl.allocate_shared_memory()` + `shm.store(x)` | Yes — seeded with real data before gl.call() | ✓ FLOWING |
| shared_accumulate_kernel | `vals` (distributed tensor) | `gl.load(x_ptr + idx)` | Yes — loads 256 float32 values from GPU memory | ✓ FLOWING |
| shared_accumulate_kernel | `shm` (shared memory) | `gl.allocate_shared_memory(gl.float32, [256])` | Starts zero (by design); populated by shared_accumulate | ✓ FLOWING |
| swizzle_kernel | `shm` (shared memory) | `gl.allocate_shared_memory(gl.float32, [32, 16], SHARED_LAYOUT)` | Populated by write_swizzled_2d with identity values | ✓ FLOWING |

All data paths are structurally complete. No hardcoded-empty return values, no disconnected props, no static fallbacks. All data flows originate from `gl.load()` (reads GPU memory), `shm.store()` (seeds shared memory), or device function writes (populate shared memory via C++ operator()).

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| TDD: shared_accumulate existence + signature | `python3 python/test/gluon/test_tt_plugin_functions.py` | ALL TESTS PASSED (5/5) | ✓ PASS |
| Code: No anti-patterns in modified files | `grep -E "TBD\|FIXME\|TODO\|return null\|return {}"` | No matches | ✓ PASS |
| Code: All 6 existing test functions unmodified | `grep -c "^def test_" python/test/gluon/test_extern_call.py` | 6 existing + 3 new = 9 total (old intact) | ✓ PASS |
| GPU: test_shared_read_write | `pytest -k test_shared_read_write` | BLOCKED (LLVM build issue) | ? SKIP |
| GPU: test_shared_accumulate | `pytest -k test_shared_accumulate` | BLOCKED (LLVM build issue) | ? SKIP |
| GPU: test_swizzle_round_trip (4 patterns) | `pytest -k test_swizzle_round_trip` | BLOCKED (LLVM build issue) | ? SKIP |
| GPU: Full regression (SHTEST-03) | `pytest python/test/gluon/test_extern_call.py -x` | BLOCKED (LLVM build issue) | ? SKIP |
| Lit: Gluon + extern-call-shared-args | `cd build && lit -v test/Gluon/ test/TritonGPU/extern-call-shared-args.mlir` | BLOCKED (LLVM build issue) | ? SKIP |

### Probe Execution

No probes declared for this phase. Phase is verification-only (no migration or tooling changes).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SHTEST-01 | 07-01, 07-02 | New E2E GPU test: shared memory allocation + read/write via gl.call() + barrier + GPU output verified on RTX 5090 | ⚠️ CODE_PRESENT_UNRUN | test_shared_read_write + test_shared_accumulate (test_extern_call.py:222-258). shared_accumulate device fn (tt_plugin.cu:227-235). Blocked by LLVM build. |
| SHTEST-02 | 07-01, 07-02 | Swizzle-correctness test: non-trivial swizzled layout round-trip with offset/block bases exercised independently | ⚠️ CODE_PRESENT_UNRUN | test_swizzle_round_trip with 4 patterns (test_extern_call.py:307-342). write_swizzled_2d device fn (tt_plugin.cu:237-242). Python evaluate_shared() reference (lines 280-304). Blocked by LLVM build. |
| SHTEST-03 | 07-02 | All 6 existing extern-call tests pass unchanged; Gluon lit suite unaffected | ⚠️ CODE_PRESENT_UNRUN | Existing tests unmodified (lines 1-167). No MLIR/dialect changes. GPU + lit runs blocked by LLVM build. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | No TBD, FIXME, TODO, stub, or placeholder patterns found in Phase 7 code | — | — |

The `PlaceholderLayout` matches in tt_plugin.cu (lines 81, 92-94) are pre-existing Phase 1 infrastructure — not stubs added by this phase. The phase-added code (lines 227-242) is clean with zero debt markers.

### Human Verification Required

#### 1. SHTEST-01: E2E Shared Memory Read-Rewrite-Write

**Test:** Run `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python -m pytest python/test/gluon/test_extern_call.py::test_shared_read_write python/test/gluon/test_extern_call.py::test_shared_accumulate -x -s --tb=short`
**Expected:** Both tests pass. `test_shared_read_write`: `assert_close(out, x * 2.0)`. `test_shared_accumulate`: `assert_close(out, x)`. D-31 PTX assertions pass for both.
**Why human:** GPU execution requires a working libtriton.so build on RTX 5090. Currently blocked by pre-existing LLVM dynamic-linking issue (libLLVM.so.23.0git CLI option double-registration with installed triton's static LLVM). A build fix or separate triton environment is needed.

#### 2. SHTEST-02: Swizzle Round-Trip (4 Parametrized Patterns)

**Test:** Run `python -m pytest python/test/gluon/test_extern_call.py::test_swizzle_round_trip -v -s --tb=short`
**Expected:** All 4 patterns pass: identity, offset_only, cross_dim, full_xor. `assert_close(out, compute_swizzle_expected((32,16), offset_bases))` for each. D-31 PTX assertion passes for each.
**Why human:** Requires GPU execution on RTX 5090. Python `evaluate_shared()` replicates C++ `SharedLinearLayout::evaluate()` XOR logic exactly — bit-for-bit match expected but unconfirmed without GPU run.

#### 3. SHTEST-03: Full Regression (No Degradation)

**Test:** Run full suite:
```bash
PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python -m pytest \
  python/test/gluon/test_extern_call.py \
  python/test/gluon/test_shared_tensor.py -x -s --tb=short -n 8
```
**Expected:** All tests pass: 6 existing + 2 SHTEST-01 + 4 SHTEST-02 parametrized + 4 Phase 4 shared_tensor = 16 total passes, zero failures.

**Test:** Run lit suite:
```bash
cd build && ninja triton-opt && lit -v test/Gluon/ test/TritonGPU/extern-call-shared-args.mlir
```
**Expected:** All 6 lit tests pass (5 Gluon + 1 Phase 6 extern-call-shared-args).
**Why human:** GPU + lit execution blocked by LLVM build issue. Existing test code (lines 1-167 of test_extern_call.py) is unmodified. No MLIR/dialect changes. Regression risk is minimal but must be confirmed before v1.1 ship.

#### 4. D-31 L-01 Landmine Exercise (Failure Path)

**Test:** In `shared_read_write_kernel`, temporarily rename `gl.barrier()` to `gl.barrier_missing()` (or remove the barrier). Confirm the D-31 PTX assertion correctly fails because the shared-memory instruction disappears from PTX.
**Expected:** AssertionError with "L-01 LANDMINE: Expected ld.shared or st.shared in PTX but found neither."
**Why human:** Verifies the landmine's failure path actually triggers — confirming the assertion is not a no-op. Cannot test without GPU run.

### Gaps Summary

**No code-level gaps found.** All planned artifacts exist, are substantive, are wired, and data flows through all paths. TDD file-content tests pass. Zero anti-patterns. The phase implementation is code-complete and matches plan specifications exactly.

**The sole blocker is the pre-existing LLVM build environment issue** (`libLLVM.so.23.0git` dynamic-linking conflict with installed triton's static LLVM), which affects ALL GPU execution in the current development environment — not just Phase 7 tests. This is a project infrastructure issue documented in both SUMMARY files (07-01: "Pre-existing LLVM build issue", 07-02: "LLVM CLI option double-registration").

**Timeline context:** Both SUMMARYs document attempted fixes (CMake target replacement, `LLVM_LINK_LLVM_DYLIB`, `patchelf --remove-needed`) — all failed. Root fix requires rebuilding LLVM with consistent static-only or shared-only linking, which is beyond this phase's scope.

**Code quality assessment:** Excellent. The implementation follows an atomic TDD RED-GREEN cycle (6 commits). All code matches plan specifications character-for-character. No shortcuts or workarounds. The file-content TDD harness provides regression protection even without GPU availability.

**Recommendation:** Phase 7 should be marked **COMPLETE_WITH_KNOWN_BLOCKERS** once the human verifier confirms the GPU execution passes on a working build. The LLVM build issue should be tracked as a separate milestone-gate blocker for v1.1 ship.

---

_Verified: 2026-07-22_
_Verifier: the agent (gsd-verifier)_
