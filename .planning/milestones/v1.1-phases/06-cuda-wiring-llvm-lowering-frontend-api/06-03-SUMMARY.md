---
phase: 06-cuda-wiring-llvm-lowering-frontend-api
plan: 03
subsystem: compiler
tags: [llvm, mlir, lowering, shared-memory, addrspace, lit]

requires:
  - phase: 06-02
    provides: ttg.extern_call_arg_spaces module attribute

provides:
  - getArgMemorySpaces helper parsing ttg.extern_call_arg_spaces JSON attribute
  - Per-operand shared-vs-distributed branch bypassing alloca+store+ptr for shared operands
  - mixed-arg lit test verifying ptr<3> lowering for shared and alloca+store for distributed

affects:
  - Phase 7 GPU E2E tests

tech-stack:
  added: []
  patterns:
    - getShmemAffineBase for AS3 pointer GEP (base + subview offset in one call)
    - LLVM::getSharedMemoryObjectFromStruct for extracting memdesc struct fields
    - Module-attribute JSON parsing pattern (replicated from getMangledName/getExtractorNames)

key-files:
  created:
    - test/TritonGPU/extern-call-shared-args.mlir
  modified:
    - lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp

key-decisions:
  - "Used getShmemAffineBase (one call) instead of explicit getShmemOffset GEP — cleaner for single-base non-partitioned shared memory"
  - "argSpaces empty fallback assigns all-register — graceful degradation for pre-Phase-6 or tensor-only modules"
  - "LLVM:: namespace qualifier required for getSharedMemoryObjectFromStruct (defined in LLVM namespace in header, unlike getShmemAffineBase which is a member function)"

patterns-established:
  - "getArgMemorySpaces: lenient JSON-parsing helper returning success() on all missing-data cases"
  - "Per-operand branching driven by explicit arg_spaces module attribute, not type inspection"
  - "MLIR LLVM dialect uses !llvm.ptr<3> syntax for addrspace(3) in FileCheck patterns"

requirements-completed:
  - SHLOWER-01
  - SHLOWER-02

coverage:
  - id: D1
    description: "Shared-memory extern_call operands lower to ptr<3> passed directly to callee — bypassing alloca+store+ptr path (SHLOWER-01)"
    requirement: SHLOWER-01
    verification:
      - kind: unit
        ref: test/TritonGPU/extern-call-shared-args.mlir#SHLOWER-01-ptr3-check
        status: pass
    human_judgment: false
  - id: D2
    description: "Distributed extern_call operands keep existing alloca+store+ptr path in mixed argument lists (SHLOWER-01 regression)"
    requirement: SHLOWER-01
    verification:
      - kind: unit
        ref: test/TritonGPU/extern-call-shared-args.mlir#SHLOWER-01-alloca-check
        status: pass
    human_judgment: false
  - id: D3
    description: "Subview offset GEP applied to shared base — callee receives base + shmemOffset, not raw allocation base (SHLOWER-02)"
    requirement: SHLOWER-02
    verification:
      - kind: unit
        ref: test/TritonGPU/extern-call-shared-args.mlir#SHLOWER-02-gep-check
        status: pass
    human_judgment: false

duration: 4 min
completed: 2026-07-15
status: complete
---

# Phase 06 Plan 03: Shared-Memory LLVM Lowering Summary

**Shared-memory `ttg.extern_call` operands lower to `!llvm.ptr<3>` via `getShmemAffineBase` bypassing alloca, while distributed operands keep the existing `alloca+store+ptr` path — verified by mixed-arg lit test**

## Performance

- **Duration:** 4 min
- **Started:** 2026-07-15T19:37:19Z
- **Completed:** 2026-07-15T19:41:09Z
- **Tasks:** 3
- **Files modified:** 1
- **Files created:** 1

## Accomplishments
- Added `getArgMemorySpaces` helper parsing per-symbol per-arg memory spaces from `ttg.extern_call_arg_spaces` JSON module attribute — lenient fallback returns `success()` with empty spaces for pre-Phase-6/tensor-only modules
- Replaced uniform `alloca+store+ptr` loop with per-operand branch: shared operands extract `SharedMemoryObject` from promoted memdesc struct, apply subview offset via `getShmemAffineBase`, and pass `ptr addrspace(3)` directly to callee (SHLOWER-01/02)
- Created `extern-call-shared-args.mlir` lit test verifying `!llvm.ptr<3>` for shared, `llvm.alloca`+`llvm.store` for distributed, and GEP for subview offset — all 3 lit tests pass (new + 2 Phase 5 regression)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add getArgMemorySpaces helper** - `11c935e3a5` (feat)
2. **Task 2: Add per-operand shared-vs-distributed branch** - `2d9794f25c` (feat)
3. **Task 3: Create mixed-arg lowering lit test** - `99c2f26bb5` (test)

**Plan metadata:** To be committed with docs

## Files Created/Modified
- `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` - Added `getArgMemorySpaces` helper (30 lines) + per-operand shared-vs-distributed branch replacing uniform alloca loop (+26/-8 lines)
- `test/TritonGPU/extern-call-shared-args.mlir` - New lit test (43 lines) with mixed shared+distributed `ttg.extern_call` lowering through `convert-triton-gpu-to-llvm` via FileCheck

## Decisions Made
- Used `getShmemAffineBase` (one call combines base extraction + `getShmemOffset` GEP) instead of explicit two-step `getSharedMemoryObjectFromStruct` + manual GEP — cleaner for single-base non-partitioned shared memory (A2 assumption holds)
- `argSpaces` empty fallback assigns all-register — graceful degradation for pre-Phase-6 or tensor-only modules
- `LLVM::` namespace qualifier required for `getSharedMemoryObjectFromStruct` (declared in `mlir::LLVM` namespace in Utility.h)
- MLIR LLVM dialect uses `!llvm.ptr<3>` syntax (not `ptr addrspace(3)`) — FileCheck patterns adjusted accordingly

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

None — all functionality is implemented end-to-end within this plan's scope.

## Threat Flags

None — no new trust boundaries, network endpoints, or schema changes beyond what was already scoped in the threat model.

## Next Phase Readiness

- Phase 06 Plan 03 is the final plan of Phase 06 — all 3 plans complete
- SHLOWER-01/02 lit test verification in place (3/3 lit tests pass)
- Ready for Phase 7 GPU E2E tests (swizzle round-trip, shared-memory functional correctness)
- Per D-23: manual LLVM IR dump inspection for subview offset GEP correctness deferred to Phase 7

---
*Phase: 06-cuda-wiring-llvm-lowering-frontend-api*
*Completed: 2026-07-15*
