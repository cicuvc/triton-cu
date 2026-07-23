---
phase: 06-cuda-wiring-llvm-lowering-frontend-api
plan: "02"
subsystem: compiler
tags: [cuda, shared-memory, clang, addrspace, mlir, module-attribute]

# Dependency graph
requires:
  - phase: 05-mlir-op-relaxation-spec-extraction
    provides: Phase-5 spec JSON with memory_space/offset_bases/block_bases keys
  - phase: 04-c-templates-clang-ast-foundation
    provides: SharedTensorParameter pybind, TypeBuilder::BuildSharedTensor, TypeInspector
provides:
  - infer_result() SharedTensorParameter degenerate-basis branch for semantic-time template deduction
  - _pre_compile_extern_calls() SharedTensorParameter construction from Phase-5 specs (suspended + fallback paths)
  - ttg.extern_call_arg_spaces module attribute (per-symbol per-arg memory-space lists)
  - BuildSharedTensor LangAS::cuda_shared addrspace qualifier for ptr addrspace(3) callee signatures
affects:
  - 06-03-LLVM lowering (reads ttg.extern_call_arg_spaces for per-operand shared-vs-register branching)
  - 07-verification (E2E shared-memory tests)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "memory_space-keyed branching: Python loops branch on ap.get('memory_space') == 'shared' to build SharedTensorParameter vs TensorParameter"
    - "Module attr side-channel: Python backend writes per-symbol arg-space JSON via mod.set_str_attr; C++ lowering reads it via getAttrOfType<StringAttr> + llvm::json::parse"
    - "Degenerate-basis pattern: Semantic-time inference uses all-zero offset/block bases + default alignment for template deduction; real bases flow at llir stage from Phase-5 specs"

key-files:
  created: []
  modified:
    - third_party/nvidia/backend/compiler.py (infer_result shared branch, _pre_compile_extern_calls spec loops, arg_spaces_map metadata, make_llir module attr emission)
    - python/src/clang_compiler.cc (BuildSharedTensor LangAS::cuda_shared + LValueReferenceType; #include <clang/Basic/AddressSpaces.h>)

key-decisions:
  - "D-12: infer_result() uses degenerate all-zero SharedTensorParameter for template deduction — only T + Shape matter at semantic time; real bases flow at llir stage"
  - "D-15: BuildSharedTensor applies LangAS::cuda_shared addrspace qualifier + LValueReferenceType so compiled bitcode emits ptr addrspace(3) natively"
  - "D-16: _pre_compile_extern_calls() builds arg_spaces_map from spec JSON and emits ttg.extern_call_arg_spaces StringAttr for Plan 06-03's C++ lowering"

patterns-established:
  - "memory_space-keyed Python branching: ap.get('memory_space') == 'shared' → SharedTensorParameter, else → TensorParameter"
  - "Module attribute JSON side-channel: Python set_str_attr + json.dumps → C++ getAttrOfType + llvm::json::parse"
  - "Clang addrspace qualifier: Ctx.getLValueReferenceType(Ctx.getAddrSpaceQualType(type, LangAS::cuda_shared))"

requirements-completed: [SHWIRE-01]

# Coverage metadata
coverage:
  - id: D1
    description: "infer_result() builds degenerate SharedTensorParameter for shared args with memory_space discriminator — template deduction succeeds with all-zero offset/block bases"
    requirement: SHWIRE-01
    verification:
      - kind: unit
        ref: "grep -c 'ap.get(\"memory_space\") == \"shared\"' third_party/nvidia/backend/compiler.py → 1; grep -c 'llvm.SharedTensorParameter()' → 3; grep -c 'stp.offset_basis = []' → 1"
        status: pass
    human_judgment: false
  - id: D2
    description: "Both suspended and fallback spec-consumption loops in _pre_compile_extern_calls() build SharedTensorParameter for shared inputs — single-parse assertion holds per SHWIRE-01"
    requirement: SHWIRE-01
    verification:
      - kind: unit
        ref: "grep -c 'inp.get(\"memory_space\") == \"shared\"' third_party/nvidia/backend/compiler.py → 2 (both paths); py_compile syntax check exits 0"
        status: pass
    human_judgment: false
  - id: D3
    description: "ttg.extern_call_arg_spaces module attribute is populated from spec JSON and emitted in make_llir() — Plan 06-03 lowering can read per-operand memory spaces"
    requirement: SHWIRE-01
    verification:
      - kind: unit
        ref: "grep -c '\"extern_call_arg_spaces\"' third_party/nvidia/backend/compiler.py → 3 (metadata key + guard + attr name); grep -c 'ttg.extern_call_arg_spaces' → 1"
        status: pass
    human_judgment: false
  - id: D4
    description: "BuildSharedTensor returns LValueReferenceType wrapping an addrspace-qualified SharedTensor type — compiled bitcode callee signature natively takes ptr addrspace(3)"
    requirement: SHWIRE-01
    verification:
      - kind: unit
        ref: "grep -c 'LangAS::cuda_shared' python/src/clang_compiler.cc → 2; CC=clang CXX=clang++ bash build.sh exits 0 (324/324 targets)"
        status: pass
    human_judgment: false

# Metrics
duration: 4min
completed: 2026-07-15
status: complete
---

# Phase 06 Plan 02: Shared-Memory Compilation Wiring Summary

**Wired SharedTensorParameter through infer_result degenerate-basis, _pre_compile_extern_calls spec-consumption loops, ttg.extern_call_arg_spaces module attribute, and BuildSharedTensor LangAS::cuda_shared addrspace qualifier**

## Performance

- **Duration:** 4 min
- **Started:** 2026-07-15T19:31:23Z
- **Completed:** 2026-07-15T19:35:15Z
- **Tasks:** 4
- **Files modified:** 2

## Accomplishments
- `infer_result()` branches on `memory_space == "shared"` to build degenerate `SharedTensorParameter` with all-zero offset/block bases + default alignment for semantic-time template deduction (D-12/D-14)
- `_pre_compile_extern_calls()` both suspended-compiler and fallback spec-consumption loops build `SharedTensorParameter` from Phase-5 spec JSON's `offset_bases`/`block_bases`/`alignment` fields (D-16)
- `ttg.extern_call_arg_spaces` module attribute carries per-symbol per-arg `["register"|"shared"]` lists from Python backend to Plan 06-03's C++ lowering
- `BuildSharedTensor` applies `Ctx.getLValueReferenceType(Ctx.getAddrSpaceQualType(type, LangAS::cuda_shared))` so compiled bitcode emits `ptr addrspace(3)` in callee signatures (D-15)

## Task Commits

1. **Task 1: infer_result() SharedTensorParameter degenerate-basis branch** - `c4d4c2efac` (feat)
2. **Task 2: _pre_compile_extern_calls() spec-consumption SharedTensorParameter** - `3001c3a0df` (feat)
3. **Task 3: ttg.extern_call_arg_spaces module attribute emission** - `0024b4de2f` (feat)
4. **Task 4 (RED): BuildSharedTensor LangAS::cuda_shared failing test** - `60704aa396` (test)
5. **Task 4 (GREEN): BuildSharedTensor LangAS::cuda_shared implementation** - `ffbb5cef63` (feat)

## Files Modified
- `third_party/nvidia/backend/compiler.py` — `infer_result()` shared branch (lines 270-309), `_pre_compile_extern_calls()` suspended + fallback spec loops (lines 798-844), `arg_spaces_map` metadata construction (lines 892-898), `make_llir()` module attr emission (lines 641-646)
- `python/src/clang_compiler.cc` — `#include <clang/Basic/AddressSpaces.h>`, `BuildSharedTensor()` addrspace + lvalue-reference wrapping (lines 1014-1037)

## Decisions Made
- Used `ap.get("memory_space") == "shared"` discriminator with `dict.get()` default-fallback to `TensorParameter` path (matches threat mitigation T-06-03)
- Degenerate all-zero `offset_basis = []` + `block_basis = []` in infer_result mirrors v1.0 degenerate-basis pattern for distributed args (D-12)
- `arg_spaces_map` uses `"shared" if inp.get("memory_space") == "shared" else "register"` discriminator — symmetry with the infer_result branch
- Module attribute follows existing `mod.set_str_attr` + `_json.dumps` pattern used by `extern_call_mangled_names` and `extern_call_extractor_names`

## Deviations from Plan
None — plan executed exactly as written. All four tasks completed with source assertions verified. TDD Task 4 followed RED→GREEN cycle with separate commits for test marker and implementation.

## Issues Encountered
None. Build succeeded on first attempt (324/324 targets, zero warnings from clang_compiler.cc).

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- Plan 06-03 (LLVM Lowering) can read `ttg.extern_call_arg_spaces` module attribute for per-operand shared-vs-register branching
- Phase 7 E2E validation can verify: single-parse guard holds, `ptr addrspace(3)` in LLVM IR, arg_spaces JSON shape correct
- Pre-existing LSP type errors in compiler.py's CUDAOptions dataclass are unrelated to this plan's changes

---
*Phase: 06-cuda-wiring-llvm-lowering-frontend-api*
*Completed: 2026-07-15*
