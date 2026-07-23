---
phase: "06-cuda-wiring-llvm-lowering-frontend-api"
verified: "2026-07-16T14:58:00Z"
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 6: CUDA Wiring + LLVM Lowering + Frontend API Verification Report

**Phase Goal:** Shared-memory args flow end-to-end from `gl.call()` through CUDA compilation to LLVM lowering with correct `ptr addrspace(3)` emission, and the frontend accepts `shared_memory_descriptor` arguments
**Verified:** 2026-07-16
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `_pre_compile_extern_calls()` builds `SharedTensorParameter` for shared inputs and routes them through the suspended-compiler path — the single-parse guard assertion at compiler.py:700 holds (parse count unchanged) | ✓ VERIFIED | Both suspended (L804-822) and fallback (L849-862) spec-consumption loops branch on `inp.get("memory_space") == "shared"` to build `llvm.SharedTensorParameter()`. Assertion at compiler.py:700 checks `parse_count_delta == distinct_cu`. All 3 lit tests + 10 Python tests pass (no assertion failure). |
| 2 | `gl.call()` in `_semantic.py` accepts a `shared_memory_descriptor` alongside tensor arguments — no TypeError or validation rejection | ✓ VERIFIED | `_core.py:805`: `isinstance(a, (tensor, shared_memory_descriptor))` bypasses `to_tensor()`. `_semantic.py:254`: `isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor))` passes validation. PaddedSharedLayout rejected at L267-273 with `NotImplementedError`. 10 Python tests pass with shared descriptors. |
| 3 | Emitted LLVM IR shows `ptr addrspace(3)` for shared-memory argument positions in the `call` instruction, and `ptr addrspace(0)` for tensor positions — verified via lit test | ✓ VERIFIED | Lit test `extern-call-shared-args.mlir` passes (3/3 lit tests pass). LLVM IR dump confirms: callee signature `@_Z10test_mixedPfPU7cuda_sharedf(!llvm.ptr, !llvm.ptr<3>)`, call site distributes `llvm.ptr` (AS0) for tensor, `!llvm.ptr<3>` for shared. |
| 4 | When a shared memory descriptor with accumulated subview offsets is passed, the callee receives `base + shmemOffset` (GEP-computed subview address), not the raw allocation base | ✓ VERIFIED | `ExternCallOpToLLVM.cpp:192-194`: `getSharedMemoryObjectFromStruct()` + `getShmemAffineBase()` applies subview offset GEP. LLVM IR dump shows `llvm.getelementptr %4[%7] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32` — base with offset applied. Lit test FileCheck confirms GEP presence. |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `python/triton/experimental/gluon/language/_core.py` | gl.call() arg dispatch with isinstance bypass for shared descriptors | ✓ VERIFIED | L803-808: loop with `isinstance(a, (tensor, shared_memory_descriptor))` passes through handles directly; `to_tensor()` only called for other types |
| `python/triton/experimental/gluon/language/_semantic.py` | Relaxed isinstance, PaddedSharedLayout guard, arg_params memory_space key | ✓ VERIFIED | L254: isinstance relaxed to `(ttgl.tensor, ttgl.shared_memory_descriptor)`; L265-273: PaddedSharedLayout rejection; L280-294: arg_params with `memory_space: "shared"` for descriptors |
| `third_party/nvidia/backend/compiler.py` | infer_result + _pre_compile_extern_calls shared branches, arg_spaces emission | ✓ VERIFIED | L272-283: infer_result degenerate SharedTensorParameter branch; L804-822 & L849-862: suspended+fallback spec-consumption branches; L645-647: module attr emission; L896-908: arg_spaces_map construction |
| `python/src/clang_compiler.cc` | BuildSharedTensor LangAS::cuda_shared addrspace qualifier | ✓ VERIFIED | L1033-1048: `Ctx.getAddrSpaceQualType(sharedTensorType, clang::LangAS::cuda_shared)` — post-merge fix applies addrspace directly (no reference wrapper, avoids SIGABRT) |
| `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` | getArgMemorySpaces + per-operand shared-vs-distributed branch | ✓ VERIFIED | L71-99: `getArgMemorySpaces` lenient JSON-parsing helper; L175-208: per-operand branch — shared operands bypass alloca via `getShmemAffineBase`, distributed preserves alloca+store+ptr |
| `test/TritonGPU/extern-call-shared-args.mlir` | Mixed-arg lit test with FileCheck | ✓ VERIFIED | 43-line lit test with FileCheck directives for alloca, store, getelementptr, ptr<3>, and call — passes cleanly |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_core.py:803` arg dispatch | `_semantic.py:253` isinstance | arg list passthrough — both use `(tensor/tlgl.tensor, shared_memory_descriptor/ttgl.shared_memory_descriptor)` | ✓ WIRED | Both ends accept shared descriptors; no type mismatch |
| `_semantic.py:287` arg_params memory_space | `compiler.py:272` infer_result branching | `ap.get("memory_space") == "shared"` → SharedTensorParameter | ✓ WIRED | Full end-to-end: frontend→infer_result→template deduction confirmed by passing Python tests |
| spec JSON memory_space key | `compiler.py:804/849` spec-consumption loops | `inp.get("memory_space") == "shared"` → SharedTensorParameter | ✓ WIRED | Both suspended (L804) and fallback (L849) paths branch correctly |
| `compiler.py:646` ttg.extern_call_arg_spaces | `ExternCallOpToLLVM.cpp:71` getArgMemorySpaces | `mod.set_str_attr` → `JSON.parse` → `argSpaces[i] == "shared"` | ✓ WIRED | Module attribute round-trip confirmed by lit test passing |
| `ExternCallOpToLLVM.cpp:192` getSharedMemoryObjectFromStruct | `ExternCallOpToLLVM.cpp:194` getShmemAffineBase | AS3 ptr → callee call site | ✓ WIRED | LLVM IR dump shows `!llvm.ptr<3>` flowing from GEP to call arg |
| BuildSharedTensor addrspace qualifier | LLVM IR compiled bitcode | `getAddrSpaceQualType(..., LangAS::cuda_shared)` → `ptr addrspace(3)` | ✓ WIRED | Callee signature in lit test LLVM IR: `@_Z10test_mixedPfPU7cuda_sharedf(!llvm.ptr, !llvm.ptr<3>)` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `compiler.py:804-822` param_types loop | `stp` (SharedTensorParameter) | spec JSON from Phase-5 extractExternCallSpecs | Spec JSON carries real offset_bases, block_bases, alignment | ✓ FLOWING |
| `compiler.py:897-908` arg_spaces_map | `spaces` list | `spec["inputs"]` memory_space field | Branch driven by actual spec key value | ✓ FLOWING |
| `ExternCallOpToLLVM.cpp:192-194` shared path | `smemObj` (SharedMemoryObject) | `promotedOperands[i]` from memdesc struct | Base ptr already AS3 via convertMemDescType | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Lit: mixed-arg extern_call lowering | `lit -v test/TritonGPU/extern-call-shared-args.mlir` | PASS | ✓ PASS |
| Lit: tensor-only regression | `lit -v test/TritonGPU/extern-call-tensor-only.mlir` | PASS | ✓ PASS |
| Lit: mixed-inputs regression | `lit -v test/TritonGPU/extern-call-mixed-inputs.mlir` | PASS | ✓ PASS |
| Python: extern_call E2E tests | `pytest python/test/gluon/test_extern_call.py -q` | 6 passed | ✓ PASS |
| Python: shared_tensor E2E tests | `pytest python/test/gluon/test_shared_tensor.py -q` | 4 passed | ✓ PASS |
| LLVM IR: ptr addrspace(3) emission | `triton-opt extern-call-shared-args.mlir -convert-triton-gpu-to-llvm` | `!llvm.ptr<3>` in callee signature + call site | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SHWIRE-01 | 06-02 | `_pre_compile_extern_calls()` builds SharedTensorParameter for shared inputs and routes through suspended-compiler path | ✓ SATISFIED | Both spec-consumption loops branch on `memory_space == "shared"`; single-parse assertion holds; 10 Python tests pass |
| SHLOWER-01 | 06-03 | Shared-memory operands lower to `ptr addrspace(3)`; distributed operands keep alloca+store+ptr | ✓ SATISFIED | Lit test passes; LLVM IR dump confirms AS3 ptr for shared + AS0 ptr for distributed in same call |
| SHLOWER-02 | 06-03 | Accumulated subview offsets applied via GEP | ✓ SATISFIED | `getShmemAffineBase` produces GEP; lit test confirms `llvm.getelementptr` in output; LLVM IR dump verifies offset computation |
| SHAPI-01 | 06-01 | `gl.call()` accepts `shared_memory_descriptor` alongside tensors | ✓ SATISFIED | isinstance guards in `_core.py` (L805) and `_semantic.py` (L254) accept shared descriptors; PaddedSharedLayout rejected; memory_space threaded in arg_params |

**Orphaned Requirements Check:** All 4 requirements mapped to Phase 6 in REQUIREMENTS.md are covered by plans. No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | - |

No TBD/FIXME/XXX markers, empty returns, placeholder patterns, or hardcoded empty data in any Phase-6 modified files.

### Post-Merge Regression Fix (commit 3928966b6d)

Verified both fixes are present in the codebase:

1. **`_core.py` NameError fix:** Uses direct `tensor` and `shared_memory_descriptor` names (imported at L46 and defined at L291) rather than `ttgl.tensor`/`ttgl.shared_memory_descriptor` — no undefined `ttgl` reference.

2. **`clang_compiler.cc` SIGABRT fix:** `BuildSharedTensor` at L1043-1044 applies `getAddrSpaceQualType(type, cuda_shared)` directly on the pointee — no `getLValueReferenceType` wrapping that would trigger clang's "Expressions can't have reference type" assertion.

### Gaps Summary

No gaps found. All 4 success criteria verified with automated evidence (lit tests, LLVM IR dump, Python test suite). All 4 requirement IDs accounted for. The phase goal is achieved.

---

_Verified: 2026-07-16_
_Verifier: gsd-verifier_
