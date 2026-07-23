---
phase: 05-mlir-op-relaxation-spec-extraction
plan: 02
subsystem: compiler
tags: [mlir, clang, c++, variant, shared-memory, json-serialization]

# Dependency graph
requires:
  - phase: 05-01
    provides: relaxed ttg.extern_call ODS operand constraint (TTG_MemDescType now valid)
provides:
  - TensorSpecInput/SharedSpecInput variant structs for discriminated input spec data
  - dyn_cast branching in extractExternCallSpecs() — no crash on MemDescType operands
  - Shared-layout extraction: offset_bases, block_bases, alignment via toLinearLayout(MemDescType)
  - std::visit JSON serialization emitting variant-specific keys (memory_space, offset_bases, block_bases, alignment)
affects: [phase-06-extern-call-lowering, compiler.py consumer]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "std::variant over std::optional for discriminated union (compile-time exhaustive)"
    - "if constexpr + std::is_same_v over runtime holds_alternative for variant dispatch"
    - "shared dim names ('offset'/'block') as MLIR StringAttr constants matching gluon_ir.cc"

key-files:
  created: []
  modified:
    - python/src/clang_compiler.cc - structs (1419-1445), extraction (1447-1539), serialization (1543-1617)

key-decisions:
  - "Used std::variant<TensorSpecInput, SharedSpecInput> instead of optional fields on a single struct — per D-10, cleaner type-level separation"
  - "Fully-qualified names (mlir::triton::gpu::MemDescType, mlir::triton::gpu::SharedEncodingTrait) required due to sub-namespace not covered by `using namespace mlir`"
  - "toLinearLayout(MemDescType) overload used instead of toLinearLayout(shape, encoding) — correctly handles subviews via getAllocShape().take_back(getRank())"

patterns-established:
  - "variant + if constexpr: compile-time exhaustive visitor pattern for JSON serialization with variant-specific keys"
  - "shared dim-name constants: kOffset/kBlock StringAttr — source of truth in gluon_ir.cc:238-242"

requirements-completed: [SHMLIR-02]

# Coverage metadata
coverage:
  - id: D1
    description: "TensorSpecInput/SharedSpecInput variant structs replace monolithic SpecInput"
    requirement: "SHMLIR-02"
    verification:
      - kind: unit
        ref: "grep: struct TensorSpecInput==1, struct SharedSpecInput==1, struct SpecInput==0, std::variant==1"
        status: pass
    human_judgment: false
  - id: D2
    description: "dyn_cast branching in extractExternCallSpecs() — MemDescType path extracts offset_bases/block_bases/alignment"
    requirement: "SHMLIR-02"
    verification:
      - kind: unit
        ref: "grep: cast<RankedTensorType>(operand)==0, dyn_cast<MemDescType>==1, SharedEncodingTrait==1, toLinearLayout(memDescTy)==1"
        status: pass
    human_judgment: false
  - id: D3
    description: "std::visit JSON serialization emits memory_space/offset_bases/block_bases/alignment for shared inputs"
    requirement: "SHMLIR-02"
    verification:
      - kind: unit
        ref: "grep: std::visit==1, if constexpr==1, memory_space==1, offset_bases==1, block_bases==1"
        status: pass
    human_judgment: false
  - id: D4
    description: "Build succeeds and all 6 existing test_extern_call.py tests pass (tensor regression)"
    requirement: "SHMLIR-02"
    verification:
      - kind: e2e
        ref: "bash build.sh (exit 0) + pytest python/test/gluon/test_extern_call.py -x -v (6/6 PASSED)"
        status: pass
    human_judgment: false

# Metrics
duration: 6min
completed: 2026-07-15
status: complete
---

# Phase 05 Plan 02: Variant-based Spec Extraction with Shared-Layout Support

**Variant-based data model with dyn_cast branch + std::visit serialization — eliminates crash on MemDescType operands and emits shared-layout JSON for Phase 6 consumption**

## Performance

- **Duration:** 6 min
- **Started:** 2026-07-15T15:08:03Z
- **Completed:** 2026-07-15T15:14:10Z
- **Tasks:** 4 (3 implementation + 1 verification)
- **Files modified:** 1

## Accomplishments

- Replaced monolithic `SpecInput` struct with `TensorSpecInput` (distributed layout) and `SharedSpecInput` (shared layout) variant types in `ExternCallSpec::inputs`
- Added `dyn_cast` branching in `extractExternCallSpecs()` — tensor path uses `dyn_cast<RankedTensorType>`, shared path uses `dyn_cast<mlir::triton::gpu::MemDescType>` with `toLinearLayout(MemDescType)` for correct subview handling
- Shared branch extracts offset_bases/block_bases via `flattenBases(ll.getBases().lookup(kOffset/kBlock))` and alignment via `SharedEncodingTrait::getAlignment()`
- Updated `tritonExtractExternCallSpecs()` JSON serialization with `std::visit` + `if constexpr` — tensor output unchanged (num_warps, reg_bases, lane_bases, warp_bases), shared output adds memory_space, offset_bases, block_bases, alignment
- All 6 existing `test_extern_call.py` tests pass — tensor-only path is regression-free

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace SpecInput structs with variant model** - `f6a9c216cc` (feat)
2. **Task 2: Add MemDescType branch in extractExternCallSpecs()** - `5996def73c` (feat)
3. **Task 3: Update tritonExtractExternCallSpecs() JSON with std::visit** - `6b1c6e3e40` (feat)
4. **Task 4: Build + verify compilation + tensor regression check** — no code changes (verification-only)

## Files Created/Modified

- `python/src/clang_compiler.cc` — Core extraction/serialization file:
  - Lines 1419-1445: `TensorSpecInput` and `SharedSpecInput` structs (replacing `SpecInput`), `ExternCallSpec::inputs` uses `std::variant`
  - Lines 1447-1539: `extractExternCallSpecs()` with `dyn_cast` branching, `mapDtype` lambda, `flattenBases` lambda, shared branch with `toLinearLayout(MemDescType)`
  - Lines 1543-1617: `tritonExtractExternCallSpecs()` with `std::visit` + `if constexpr` JSON dispatch

## Decisions Made

- Used `std::variant<TensorSpecInput, SharedSpecInput>` instead of optional fields on a single struct — per D-10, type-level separation is cleaner and exhaustiveness is enforced at compile time
- `toLinearLayout(MemDescType)` overload chosen over `toLinearLayout(shape, encoding)` — correctly handles subviews via `getAllocShape().take_back(getRank())` (RESEARCH.md Pitfall 2)
- Shared dim-name constants (`kOffset`/`kBlock`) as MLIR `StringAttr` — source of truth confirmed in `gluon_ir.cc:238-242`
- `if constexpr` + `std::is_same_v` for variant dispatch — compile-time exhaustive, no runtime `holds_alternative` needed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fully-qualified names needed for MLIR sub-namespace types**
- **Found during:** Task 2 (MemDescType branch)
- **Issue:** `MemDescType` and `SharedEncodingTrait` are in `mlir::triton::gpu` namespace, not `mlir`. The `using namespace mlir` at line 1448 doesn't cover sub-namespaces. LSP reported `Unknown type name 'MemDescType'` and `Unknown type name 'SharedEncodingTrait'`.
- **Fix:** Used fully-qualified names: `mlir::triton::gpu::MemDescType` and `mlir::triton::gpu::SharedEncodingTrait`.
- **Files modified:** `python/src/clang_compiler.cc` (2 lines in shared branch)
- **Verification:** LSP errors resolved, build passes (324/324 steps)
- **Committed in:** `5996def73c` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary for correctness — MLIR namespace hierarchy requires full qualification. No scope creep.

## Issues Encountered

None — plan executed with only the expected namespace qualification adjustment.

## Known Stubs

None — all extracted data structures are fully populated from MLIR operands.

## Threat Flags

None — no new security surface introduced. Existing trust boundaries (MLIR module → extractExternCallSpecs, extractExternCallSpecs → JSON, JSON → Python consumer) are the same as before. The only change is adding a shared-memory code path that follows the same patterns.

## Next Phase Readiness

- **SHMLIR-02 satisfied:** `extractExternCallSpecs()` no longer crashes on `MemDescType` operands — `dyn_cast` branching with full shared-layout extraction is in place
- **JSON discriminator ready:** Shared inputs emit `"memory_space": "shared"` — Phase 6 consumer at `compiler.py:786-795` can use `inp.get("memory_space")` to distinguish shared vs. tensor inputs
- **Tensor backward compatibility:** Existing Phase 5 consumer uses `.get()` with defaults — new shared-input keys are silently ignored, no crash
- **Phase 6 unblocked:** `offset_bases`, `block_bases`, `alignment` now flow through the JSON pipeline for downstream lowering consumption

---

*Phase: 05-mlir-op-relaxation-spec-extraction*
*Completed: 2026-07-15*
