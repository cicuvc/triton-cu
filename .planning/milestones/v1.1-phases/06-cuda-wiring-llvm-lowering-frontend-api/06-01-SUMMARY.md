---
phase: 06-cuda-wiring-llvm-lowering-frontend-api
plan: 01
subsystem: frontend-api
tags: [gluon, gl.call, shared-memory-descriptor, arg-dispatch, isinstance, PaddedSharedLayout, memory-space, arg-params]

# Dependency graph
requires:
  - phase: 05-mlir-op-relaxation-spec-extraction
    provides: mixed tensor+memdesc extern_call operands, SharedSpecInput in spec JSON
provides:
  - gl.call() accepts shared_memory_descriptor arguments alongside tensors (no TypeError)
  - Frontend isinstance guard relaxed to accept shared_memory_descriptor
  - PaddedSharedLayout rejected at the frontend with NotImplementedError (D-19/D-20)
  - arg_params carries memory_space: "shared" key for shared descriptor args (D-14/D-21)
affects: [06-02 cuda-wiring, 06-03 llvm-lowering, 07-verify]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Frontend isinstance guards with explicit tuple (ttgl.tensor, ttgl.shared_memory_descriptor) for type safety"
    - "memory_space discriminator (absent='register', present='shared') in arg_params dict"
    - "NotImplementedError for unsupported layout types matching the existing f64 guard pattern"

key-files:
  created: []
  modified:
    - python/triton/experimental/gluon/language/_core.py
    - python/triton/experimental/gluon/language/_semantic.py

key-decisions:
  - "gl.call() arg dispatching uses isinstance bypass for both ttgl.tensor and ttgl.shared_memory_descriptor, preserving .handle attribute passthrough"
  - "PaddedSharedLayout is explicitly rejected at the frontend with NotImplementedError; SharedLinearLayout, SwizzledSharedLayout, and NVMMASharedLayout are all accepted"
  - "arg_params uses an explicit memory_space key ('shared' for descriptors, absent for distributed tensors) to signal the backend"

patterns-established:
  - "isinstance guard for multiple accepted types: isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor))"
  - "Frontend layout rejection: NotImplementedError with descriptive message naming accepted alternatives and provenance reference"
  - "Discriminator-based arg_params: presence of 'memory_space' key signals shared vs. register"

requirements-completed: [SHAPI-01]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "gl.call() accepts shared_memory_descriptor arguments without TypeError — isinstance bypass in _core.py:803 bypasses to_tensor, and _semantic.py:253-255 relaxed isinstance check passes descriptors through"
    requirement: SHAPI-01
    verification:
      - kind: unit
        ref: "grep assertion: isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor)) exists both in _core.py and _semantic.py"
        status: pass
      - kind: unit
        ref: "smoke import: PYTHONPATH=$PWD/python:$PWD/third_party/nvidia python3 -c 'from triton.experimental.gluon.language import shared_memory_descriptor'"
        status: pass
    human_judgment: false
  - id: D2
    description: "PaddedSharedLayout descriptors are rejected at the frontend with NotImplementedError matching the f64 guard style"
    requirement: SHAPI-01
    verification:
      - kind: unit
        ref: "grep assertion: PaddedSharedLayout appears in _semantic.py with isinstance check and NotImplementedError raise"
        status: pass
    human_judgment: false
  - id: D3
    description: "arg_params for shared args carry memory_space: 'shared' key, absent for distributed args"
    requirement: SHAPI-01
    verification:
      - kind: unit
        ref: "grep assertion: '\"memory_space\"' and '\"shared\"' both appear in _semantic.py arg_params construction"
        status: pass
    human_judgment: false

# Metrics
duration: 1min
completed: 2026-07-15
status: complete
---

# Phase 06 Plan 01: Frontend API — gl.call() Shared Memory Descriptor Support Summary

**Enable gl.call() to accept shared_memory_descriptor arguments alongside tensors with frontend isinstance relaxation, PaddedSharedLayout rejection guard, and memory_space-keyed arg_params signaling**

## Performance

- **Duration:** 1 min
- **Started:** 2026-07-15T19:27:48Z
- **Completed:** 2026-07-15T19:29:09Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `gl.call()` arg dispatching in `_core.py` bypasses `to_tensor()` for `shared_memory_descriptor` args — descriptors reach `call_extern()` with `.handle` intact, no `TypeError` (D-18)
- `call_extern()` isinstance check in `_semantic.py` relaxed from `ttgl.tensor` to `(ttgl.tensor, ttgl.shared_memory_descriptor)` with updated error message (D-18)
- `PaddedSharedLayout` rejected at the frontend with `NotImplementedError` listing accepted alternatives: `SharedLinearLayout`, `SwizzledSharedLayout`, `NVMMASharedLayout` (D-19/D-20)
- `arg_params` construction branches on `isinstance(a, ttgl.shared_memory_descriptor)`, appending `"memory_space": "shared"` key for shared descriptors — absent for distributed args signals register (D-14/D-21)

## Task Commits

Each task was committed atomically:

1. **Task 1: Relax gl.call() arg dispatching to bypass to_tensor for shared_memory_descriptor** - `2d98b41ee3` (feat)
2. **Task 2: Relax call_extern() isinstance, add PaddedSharedLayout guard, thread memory_space** - `27c8a2e2c7` (feat)

## Files Created/Modified

- `python/triton/experimental/gluon/language/_core.py` — `gl.call()` arg dispatching loop (line 803): isinstance bypass for `shared_memory_descriptor` replaces `to_tensor` list comprehension
- `python/triton/experimental/gluon/language/_semantic.py` — `call_extern()` method (lines 253-294): relaxed isinstance check, PaddedSharedLayout rejection guard, memory_space-keyed arg_params

## Decisions Made

None — followed plan exactly as specified in 06-01-PLAN.md and 06-PATTERNS.md.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `gl.call()` frontend ready for `shared_memory_descriptor` args with proper rejection of unsupported `PaddedSharedLayout`
- `arg_params` with `memory_space` key ready for consumption by Plan 06-02 (`InferExternCallResult.infer_result()` in `compiler.py`)
- Downstream plans (06-02 cuda-wiring, 06-03 llvm-lowering) can consume the guarded types and memory_space discriminator

---
*Phase: 06-cuda-wiring-llvm-lowering-frontend-api*
*Completed: 2026-07-15*
