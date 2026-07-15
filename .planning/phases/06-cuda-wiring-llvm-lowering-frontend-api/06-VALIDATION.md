---
phase: 6
slug: cuda-wiring-llvm-lowering-frontend-api
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-16
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | LLVM lit (FileCheck) + pytest |
| **Config file** | `test/lit.cfg.py` (lit), `pyproject.toml` (pytest/ruff/yapf/mypy) |
| **Quick run command** | `cd build && ninja triton-opt && lit -v test/TritonGPU/extern-call-shared-args.mlir` |
| **Full suite command** | `cd build && make test-lit` |
| **Estimated runtime** | ~10s (quick), ~60s (full lit suite) |

---

## Sampling Rate

- **After every task commit:** Run `cd build && ninja triton-opt && lit -v test/TritonGPU/extern-call-shared-args.mlir` (once lit test exists — Wave 0)
- **After every plan wave:** Run `cd build && make test-lit` (regression check: Phase 5 lit tests must stay green)
- **Before `/gsd-verify-work`:** Full suite must be green + parse-count assertion verified
- **Max feedback latency:** 30 seconds (ninja build + lit)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | SHAPI-01 | T-06-01 / T-06-02 | unit (grep + smoke) | `grep -v '^#' python/triton/experimental/gluon/language/_core.py \| grep -c 'to_tensor(a) for a in args'` == 0; `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python3 -c "from triton.experimental.gluon.language import shared_memory_descriptor; print('OK')"` exits 0 | N/A (existing modified) | ⬜ pending |
| 06-01-02 | 01 | 1 | SHAPI-01 | T-06-01 | unit (grep + smoke) | `grep -v '^#' python/triton/experimental/gluon/language/_semantic.py \| grep -c 'isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor))'` ≥ 1; `grep -c 'PaddedSharedLayout' python/triton/experimental/gluon/language/_semantic.py` ≥ 1; `grep -c '"memory_space"' python/triton/experimental/gluon/language/_semantic.py` ≥ 1 | N/A | ⬜ pending |
| 06-02-01 | 02 | 1 | SHWIRE-01 | T-06-04 | unit (grep + smoke) | `grep -c 'ap.get("memory_space") == "shared"' third_party/nvidia/backend/compiler.py` ≥ 1; `grep -c 'llvm.SharedTensorParameter()' third_party/nvidia/backend/compiler.py` ≥ 1; `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python3 -c "import llvm; p = llvm.SharedTensorParameter(); print('offset_basis' in dir(p))"` exits 0 | N/A | ⬜ pending |
| 06-02-02 | 02 | 1 | SHWIRE-01 | T-06-03 / T-06-04 | unit (grep + syntax) | `grep -c 'inp.get("memory_space") == "shared"' third_party/nvidia/backend/compiler.py` ≥ 1; `grep -c 'SharedTensorParameter()' third_party/nvidia/backend/compiler.py` ≥ 2 (infer_result + spec loops); `python3 -c "import py_compile; py_compile.compile('third_party/nvidia/backend/compiler.py', doraise=True)"` exits 0 | N/A | ⬜ pending |
| 06-02-03 | 02 | 1 | SHWIRE-01 | T-06-03 | unit (grep) | `grep -c '"extern_call_arg_spaces"' third_party/nvidia/backend/compiler.py` ≥ 2 (metadata key + attr name); `grep -c 'ttg.extern_call_arg_spaces' third_party/nvidia/backend/compiler.py` ≥ 1; `grep -c '"shared" if inp.get("memory_space") == "shared" else "register"' third_party/nvidia/backend/compiler.py` = 1 | N/A | ⬜ pending |
| 06-02-04 | 02 | 1 | SHWIRE-01 | T-06-05 | build | `CC=clang CXX=clang++ bash build.sh` exits 0; `grep -c 'getAddrSpaceQualType' python/src/clang_compiler.cc` ≥ 1; `grep -c 'LangAS::cuda_shared' python/src/clang_compiler.cc` ≥ 1 | N/A | ⬜ pending |
| 06-03-01 | 03 | 1 | SHLOWER-01 | T-06-06 | build | `ninja triton-opt` exits 0 (from build dir); `grep -c 'getArgMemorySpaces' lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` ≥ 1 | N/A | ⬜ pending |
| 06-03-02 | 03 | 1 | SHLOWER-01 / SHLOWER-02 | T-06-06 / T-06-07 / T-06-08 | lit + build | `cd build && ninja triton-opt && lit -v test/TritonGPU/extern-call-shared-args.mlir` passes; `lit -v test/TritonGPU/extern-call-mixed-inputs.mlir test/TritonGPU/extern-call-tensor-only.mlir` passes (regression) | ❌ W0 | ⬜ pending |
| 06-03-03 | 03 | 1 | SHLOWER-01 / SHLOWER-02 | T-06-08 | lit | File exists + contains `CHECK: ptr addrspace(3)` + `CHECK: llvm.alloca` + `CHECK: llvm.call`; `cd build && lit -v test/TritonGPU/extern-call-shared-args.mlir` passes | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `test/TritonGPU/extern-call-shared-args.mlir` — mixed shared+distributed args lowering lit test with FileCheck (created by Plan 06-03 Task 3; exercised by 06-03-02/06-03-03 verification)
- [ ] `build/triton-opt` — must already exist from Phase 5 development; re-run `cd build && ninja triton-opt` if stale
- [ ] `PYTHONPATH` smoke environment — Python import checks use the AGENTS.md canonical invocation: `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python3 -c "..."`

*No new framework installs — lit, FileCheck, and pytest are already configured from Phases 4-5.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Subview offset GEP produces correct `base + shmemOffset` in LLVM IR | SHLOWER-02 | D-23: No automated matcher for correct GEP offset arithmetic — FileCheck can confirm a GEP exists but not that the offset value is correct | After `lit -v test/TritonGPU/extern-call-shared-args.mlir` passes, run `triton-opt -convert-triton-gpu-to-llvm test/TritonGPU/extern-call-shared-args.mlir` and visually inspect the GEP instruction's offset operand matches the `shared_linear` offset encoding in the MLIR source. Phase 7 GPU functional test provides definitive proof. |
| AS3 pointer preservation: no `ld.generic`/`st.generic` on shared data | SHLOWER-01 | L-01 (landmine): LLVM cannot express "ptr-to-AS3 stored in AS0 memory"; PTX-level verification requires GPU-capable pipeline | Not automated in Phase 6. Phase 7 E2E test (pytest) should PTX-inspect for `ld.shared`/`st.shared` on shared-memory operands. A `ld.generic` on shared data = landmine fired. |
| Full E2E pipeline: `gl.call()` with `shared_memory_descriptor` arg produces correct GPU results | SHAPI-01 / SHWIRE-01 / SHLOWER-01 / SHLOWER-02 | Requires GPU; Phase 7 scope | `pytest python/test/gluon/test_extern_call.py` (Phase 7 extends this test file with shared-memory test cases) |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or are listed in Manual-Only Verifications
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (Tasks 06-01-01 through 06-02-04 all have grep/smoke/build automated commands; Tasks 06-03-01 through 06-03-03 have ninja/lit automated commands)
- [x] Wave 0 covers all MISSING references (`test/TritonGPU/extern-call-shared-args.mlir` created by 06-03-03)
- [x] No watch-mode flags
- [x] Feedback latency < 30s (ninja build + single lit test < 30s)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending — automated commands defined; lit test file is Wave 0 (created by Plan 06-03 Task 3)
