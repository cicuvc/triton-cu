# Phase 6: CUDA Wiring + LLVM Lowering + Frontend API - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-16
**Phase:** 6-CUDA Wiring + LLVM Lowering + Frontend API
**Areas discussed:** Semantic-time inference for shared args, addrspace(3) call convention mechanism, Frontend validation & error surface, Phase 6 verification approach

---

## Semantic-time inference for shared args

| Option | Description | Selected |
|--------|-------------|----------|
| Degenerate placeholder bases | All-zero offset/block bases + default alignment at semantic time, mirroring v1.0 distributed-arg pattern; real bases at llir stage | ✓ |
| Real bases at semantic time | Compute actual bases from the descriptor's Python SharedLayout; adds conversion plumbing | |
| PlaceholderSharedLayout fallback | C++ placeholder type like v1.0's PlaceholderLayout for pinned-layout functions | |

**User's choice:** Degenerate placeholder bases (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Layout template-deduced | Device fns use `template<class L> f(SharedTensor<T,S,L>&)`; degenerate bases deduce fine; document deducibility requirement | ✓ |
| Layout is pinned/concrete | Fn hardcodes a SharedLinearLayout; degenerate bases fail SFINAE | |
| Both, with fallback retry | Degenerate first, retry with real bases on substitution failure | |

**User's choice:** Layout template-deduced (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Add memory_space key | arg_params dict gains `memory_space: "shared"`; hook branches on it; matches Phase-5 spec JSON | ✓ |
| Separate param list | Parallel shared_arg_params list | |
| Positional convention | Shared args always first/last; fragile | |

**User's choice:** Add memory_space key

---

## addrspace(3) call convention mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| clang addr_space attribute | TypeBuilder applies LangAS::cuda_shared to the SharedTensor& pointee; callee natively takes ptr addrspace(3) | ✓ |
| Generic ptr + explicit addrspacecast | Callee stays AS0; cast at call sites; relaxes success criterion 3 | |

**User's choice:** clang addr_space attribute (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Per-operand type dispatch | Branch on original operand being MemDescType | (initially selected in an unreadable render; superseded) |
| Spec-JSON driven dispatch | Read memory_space from spec-derived metadata rather than MLIR operand type | ✓ |

**User's choice:** Spec-JSON driven dispatch (re-asked after a rendering issue; this answer stands)

| Option | Description | Selected |
|--------|-------------|----------|
| New module attribute | `_pre_compile_extern_calls()` writes per-symbol arg memory spaces (e.g. `ttg.extern_call_arg_spaces`); lowering parses like getMangledName | ✓ |
| Re-extract at lowering | Call extractExternCallSpecs again in the C++ pass | |
| Revert to type dispatch | Use isa<MemDescType> directly | |

**User's choice:** New module attribute (Recommended)

**Notes (landmine, user-reported):** From prior LLVM experience — storing a `ptr addrspace(3)` to an AS0 memory slot and reloading it can erase the AS3 tag (LLVM cannot express "pointer-to-AS3 stored in AS0"), degrading `ld.shared` to `ld.generic`. A complete fix needs MemorySSA-class machinery — explicitly out of scope for this phase, but the user wants it recorded so implementers avoid store/reload round-trips for shared pointers and Phase-7 verification checks PTX for `ld.generic` degradation. Recorded as L-01 in CONTEXT.md.

---

## Frontend validation & error surface

| Option | Description | Selected |
|--------|-------------|----------|
| Relax isinstance + frontend guards | Accept (tensor, shared_memory_descriptor); f64 + PaddedSharedLayout guards at the frontend | ✓ |
| Relax only, validate in backend | Minimal frontend change; errors far from user call site | |

**User's choice:** Relax isinstance + frontend guards (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Reject only Padded | SharedLinear/Swizzled/NVMMA all accepted (all convert via toLinearLayout, Phase-5 validated) | ✓ |
| Accept only SharedLinearLayout | More conservative; asymmetric with SHMLIR-02 | |

**User's choice:** Reject only Padded (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Same style as f64 guard | What's unsupported + alternatives + provenance | ✓ |
| Brief error | Short TypeError only | |

**User's choice:** Same style as f64 guard (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Pass layout object | arg_params carries the descriptor's Python SharedLayout in the existing layout field; backend extracts bases | ✓ |
| Pass pre-built SharedTensorParameter | Frontend constructs the pybind object itself | |

**User's choice:** Pass layout object (Recommended)

---

## Phase 6 verification approach

| Option | Description | Selected |
|--------|-------------|----------|
| Lit-only compile-tier tests | One lit test for ExternCallOpToLLVM (memdesc → addrspace(3) + GEP in LLVM dialect IR); no GPU | ✓ |
| Lit + GPU smoke test | Also assert addrspace(3)/ld.shared in dumped IR/PTX; duplicates Phase 7 | |
| Manual only | All automation in Phase 7 | |

**User's choice:** Lit-only compile-tier tests (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Single mixed-arg lit | One file: distributed positions alloca+store+ptr, shared positions ptr addrspace(3)+GEP; validates ordering too | ✓ |
| Two lit tests | Separate addrspace + subview files | |
| One file, both scenarios | Add a subslice function to the same file | |

**User's choice:** Single mixed-arg lit (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Manual now, GPU-proven in P7 | Criterion 4 (subview GEP) manually inspected in Phase 6; Phase 7 swizzle test proves functionally | ✓ |
| Fold into the lit file | Lit-check criterion 4 too | |

**User's choice:** Manual now, GPU-proven in P7 (Recommended)

---

## The agent's Discretion

- Exact module-attribute name (`ttg.extern_call_arg_spaces` suggested) and JSON shape
- `getShmemAffineBase` vs `getSharedMemoryObjectFromStruct`+`getShmemOffset` in the shared branch
- `promoteOperands` interaction with memdesc operands
- SharedLayout→(offset_bases, block_bases, alignment) extraction helper structure in the Python hook

## Deferred Ideas

- AS3 pointer preservation across store/reload (MemorySSA-class analysis) — recorded as landmine L-01 only
- PlaceholderSharedLayout fallback for pinned-layout device functions — revisit if real users need it
