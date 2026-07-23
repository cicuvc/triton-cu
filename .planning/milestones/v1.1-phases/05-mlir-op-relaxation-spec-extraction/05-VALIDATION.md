---
phase: 5
slug: mlir-op-relaxation-spec-extraction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-15
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MLIR lit tests (llvm-lit) + pytest |
| **Config file** | `test/lit.cfg.py` |
| **Quick run command** | `cd $BUILD_DIR && ninja triton-opt && lit -v test/TritonGPU/<test>.mlir` |
| **Full suite command** | `make test-lit` (from build dir) |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run the quick run command (targeted lit test)
- **After every plan wave:** Run `make test-lit`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| _(filled by planner)_ | | | SHMLIR-01, SHMLIR-02 | — | N/A | lit | `lit -v test/...` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Lit test file for mixed tensor+memdesc `ttg.extern_call` — stubs for SHMLIR-01
- [ ] Spec-extraction check (JSON keys `memory_space`/`offset_bases`/`block_bases`/`alignment`) — SHMLIR-02

*Existing lit infrastructure covers the harness; new test files required.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
