---
phase: 07
slug: e2e-verification
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-21
---

# Phase 07 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing, no config file changes) |
| **Config file** | No separate config — `pytestmark` in `test_extern_call.py` + `__init__.py` |
| **Quick run command** | `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python -m pytest python/test/gluon/test_extern_call.py -x -s --tb=short -k "<test_name>"` |
| **Full suite command** | `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python -m pytest python/test/gluon/test_extern_call.py python/test/gluon/test_shared_tensor.py -x -n 8` |
| **Estimated runtime** | ~30 seconds (GPU tests) + ~5 seconds (lit tests) |

---

## Sampling Rate

- **After every task commit:** Run `pytest python/test/gluon/test_extern_call.py -x --tb=short -k "<test_name>"` (single test)
- **After every plan wave:** Full GPU test suite: `pytest python/test/gluon/test_extern_call.py python/test/gluon/test_shared_tensor.py -x -n 8`
- **Before `/gsd-verify-work`:** Full suite must be green + all lit tests pass
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | SHTEST-01, SHTEST-02 | T-07-01, T-07-02 | Loop bound = REG_SIZE prevents out-of-bounds shared writes | compile-check (parse-only) | `PYTHONPATH=... python3 -c "..."` (see PLAN.md verify) | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 2 | SHTEST-01 | T-07-04, T-07-05 | gl.barrier() between write and read; seeded shared memory | GPU E2E | `pytest test_extern_call.py -k "test_shared_read_write or test_shared_accumulate" -x` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 2 | SHTEST-02 | T-07-06 | Swizzle byte-offset XOR deterministic, Python reference cross-checked | GPU E2E | `pytest test_extern_call.py::test_swizzle_round_trip -x -v` | ❌ W0 | ⬜ pending |
| 07-02-03 | 02 | 2 | SHTEST-03 | T-07-08 | Fixed-shape tensors, single-CTA, no dynamic allocation | GPU E2E + lit regression | `pytest test_extern_call.py test_shared_tensor.py -x -n 8 && cd build && ninja triton-opt && lit -v test/Gluon/ test/TritonGPU/extern-call-shared-args.mlir` | ✅ (existing 6 + 4 + lit suite) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `python/test/gluon/test_extern_call.py` — test functions to add:
  - `test_shared_read_write` — SHTEST-01 sequential
  - `test_shared_accumulate` — SHTEST-01 mixed args
  - `test_swizzle_round_trip` — SHTEST-02 parametrized (4 patterns)
- [ ] `python/test/gluon/tt_plugin.cu` — CUDA device functions to add:
  - `shared_accumulate` — SHTEST-01 mixed args
  - `write_swizzled_2d` — SHTEST-02 swizzle
- [ ] D-31 PTX grep assertions — embedded in SHTEST-01/02 test bodies (no separate test file needed)
- [ ] No new test framework config needed (reuses existing pytest infrastructure)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| D-30 full regression pass | SHTEST-03 | Requires GPU environment (RTX 5090); no CI self-test possible | Run `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python -m pytest python/test/gluon/test_extern_call.py python/test/gluon/test_shared_tensor.py -x -n 8` and `cd build && ninja triton-opt && lit -v test/Gluon/ test/TritonGPU/extern-call-shared-args.mlir` — all 16 tests pass |
| D-31 PTX grep for ld.shared/st.shared | SHTEST-01, SHTEST-02 | Embedded assertion in each test body; no separate test to run | Run SHTEST-01 and SHTEST-02 tests — assert fires within test body |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
