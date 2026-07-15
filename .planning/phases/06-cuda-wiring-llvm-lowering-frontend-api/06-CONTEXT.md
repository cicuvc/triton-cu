# Phase 6: CUDA Wiring + LLVM Lowering + Frontend API - Context

**Gathered:** 2026-07-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Shared-memory args flow end-to-end: `gl.call()` accepts `shared_memory_descriptor` arguments (SHAPI-01), `_pre_compile_extern_calls()` builds `SharedTensorParameter` from the Phase-5 shared specs through the single-parse suspended-compiler path (SHWIRE-01), and `ExternCallOpToLLVM.cpp` lowers memdesc operands as `ptr addrspace(3)` passed directly to the callee with accumulated subview offsets applied (SHLOWER-01/02).

**In scope:** frontend isinstance relaxation + guards in `_semantic.py:call_extern`; `memory_space`-keyed arg_params for the inference hook; semantic-time `SharedTensorParameter` construction in `infer_result`; llir-stage `SharedTensorParameter` construction in `_pre_compile_extern_calls` from Phase-5 spec JSON; a new module attribute carrying per-symbol arg memory spaces to the C++ lowering; shared-operand branch in `ExternCallOpToLLVM.cpp`; clang address-space qualifier in `TypeBuilder::BuildSharedTensor`; one mixed-arg lit test.

**Explicitly NOT this phase:** E2E GPU correctness + swizzle round-trip tests (Phase 7, SHTEST-01/02/03); shared-memory return types (SHRET-01, future); PaddedSharedLayout support; auto-inserted barriers; AS3-pointer-preservation-across-store/reload machinery (recorded as landmine only). Covers requirements **SHWIRE-01, SHLOWER-01, SHLOWER-02, SHAPI-01**.

</domain>

<decisions>
## Implementation Decisions

### Semantic-time inference for shared args (SHWIRE-01 semantic half)
- **D-12:** At semantic time, `infer_result()` builds `SharedTensorParameter` with **degenerate all-zero offset/block bases + default alignment** — mirrors the v1.0 degenerate-basis pattern for distributed args (`compiler.py:276-295`). Template deduction only needs T + Shape to resolve the return type; real bases flow at the llir stage from Phase-5 extracted specs. No Python-side layout math at semantic time.
- **D-13:** Phase 6 assumes the **shared-layout template parameter is DEDUCED** (`template<class L> f(SharedTensor<T,S,L>&)`). Device functions that pin a concrete `SharedLinearLayout` in their signature are NOT supported by degenerate-basis inference. No `PlaceholderSharedLayout` fallback this phase — document the deducibility requirement.
- **D-14:** The inference hook distinguishes shared from distributed args via a **`memory_space: "shared"` key in the arg_params dict** (absent/`"register"` for distributed). Matches the Phase-5 spec JSON schema, making llir-stage branching symmetric.

### addrspace(3) call convention (SHLOWER-01)
- **D-15:** `TypeBuilder::BuildSharedTensor` applies a **clang address-space qualifier (`LangAS::cuda_shared`) to the `SharedTensor&` pointee** so the mangled callee signature natively takes `ptr addrspace(3)` — no addrspacecast at call sites. Success criterion 3 ("call inst shows ptr addrspace(3)") is met directly. Note: clang-specific, relies on the pinned LLVM's LangAS handling.
- **D-16:** In `ExternCallOpToLLVM`, the per-operand shared-vs-distributed branch is **spec-JSON driven**: `_pre_compile_extern_calls()` writes each symbol's per-arg memory-space list into a **new module attribute (e.g. `ttg.extern_call_arg_spaces`)**, parsed in the lowering via the same JSON-parsing pattern as `getMangledName`/`getExtractorNames` (`ExternCallOpToLLVM.cpp:13-69`).
- **D-17:** Shared operands bypass the distributed `alloca+store+ptr` path entirely: `getSharedMemoryObjectFromStruct` on the promoted memdesc struct → base + `getShmemOffset` GEP (or `getShmemAffineBase` which combines both, `Utility.cpp:1401`) → pass the resulting `ptr addrspace(3)` **directly** to the callee. Distributed operands keep the existing path; mixed arg lists preserve callee signature order.

### Frontend validation & error surface (SHAPI-01)
- **D-18:** `call_extern()` relaxes `isinstance(a, ttgl.tensor)` to `isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor))` (`_semantic.py:253-255`). Frontend guards (f64, unsupported layouts) live here, at the user's call site.
- **D-19:** Accepted shared layouts: **only PaddedSharedLayout is rejected** at the frontend. `SharedLinearLayout`, `SwizzledSharedLayout`, and `NVMMASharedLayout` are all accepted (all three convert via `toLinearLayout`, validated in Phase 5 / SHMLIR-02).
- **D-20:** Rejection error style matches the existing f64 guard: states what's unsupported + usable alternatives + provenance, e.g. `"gl.call() does not support PaddedSharedLayout shared memory; use SharedLinearLayout/SwizzledSharedLayout/NVMMASharedLayout (see v1.1 out-of-scope)"`.
- **D-21:** Shared-layout info reaches the backend by **passing the layout object** — arg_params carries the descriptor's `a.type.layout` (Python `SharedLayout` object) in the existing `layout` field, consistent with how distributed args pass `DistributedLayout`. The backend hook extracts `offset_bases`/`block_bases`/`alignment` from it (helper analogous to `_scalar_type_for`).

### Phase 6 verification approach
- **D-22:** **Lit-only compile-tier automation**: one new lit test for `ExternCallOpToLLVM` — a single **mixed-arg** (shared + distributed) `ttg.extern_call` run through `convert-triton-gpu-to-llvm`, FileCheck-verifying distributed positions get `alloca+store+ptr` (AS0) and shared positions get `ptr addrspace(3)`. No GPU dependency.
- **D-23:** Success criterion 4 (subview offsets via `getShmemOffset` GEP, SHLOWER-02) is verified by **manual LLVM IR dump inspection** in Phase 6; Phase 7's swizzle round-trip GPU test proves it functionally (wrong offset ⇒ wrong values read back).

### Landmine (recorded, NOT fixed this phase)
- **L-01 (user-reported, from prior LLVM experience):** LLVM cannot express "a pointer-to-AS3 stored in AS0 memory". If a `ptr addrspace(3)` is stored to a memory slot and reloaded, the AS3 tag can be erased, silently degrading `ld.shared`/`st.shared` to `ld.generic`/`st.generic` (slower). A complete fix needs MemorySSA-class machinery — **out of scope for this phase**. Mitigations now: (a) D-17 deliberately passes the AS3 pointer **directly** to the callee with no alloca/store/load round-trip; (b) implementers must NOT route shared pointers through stack slots in the lowering or in linked bitcode; (c) Phase 7 verification should eyeball the PTX for `ld.shared`/`st.shared` (a `ld.generic` on shared data = this landmine fired).

### Locked upstream (carried forward — do not re-decide)
- `SharedTensor` is argument-only for v1.1; zero-length `T data[]` base-pointer storage (D-03, Phase 4); `SharedTensor&` mutable-reference param convention (D-05, Phase 4).
- Spec JSON schema: `memory_space`/`offset_bases`/`block_bases`/`alignment` via `std::variant<TensorSpecInput, SharedSpecInput>` (D-10, Phase 5).
- Single-parse guard must hold: `compiler.py:683` assertion (parse delta == distinct .cu count) unchanged.
- No auto-barriers — user places `gl.barrier()`; PaddedSharedLayout, dynamic shared, TMA interop all out of scope (REQUIREMENTS.md).

### Agent's Discretion
- Exact module-attribute name for arg memory spaces (suggested: `ttg.extern_call_arg_spaces`), and its JSON shape (per-symbol array of `"shared"`/`"register"` strings suggested).
- Whether the shared lowering branch uses `getShmemAffineBase` (one call) or `getSharedMemoryObjectFromStruct` + explicit `getShmemOffset` GEP (two steps) — functionally identical; pick whichever reads cleaner given the promoted-operand plumbing.
- How `promoteOperands` interacts with memdesc operands (whether shared operands should skip promotion) — planner/researcher to verify against `TypeConverter::convertMemDescType` (`TypeConverter.cpp:36,56`).
- The exact helper structure for SharedLayout→(offset_bases, block_bases, alignment) extraction in the Python backend hook.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & milestone intent
- `.planning/REQUIREMENTS.md` § "CUDA Compilation Wiring", "LLVM Lowering", "Frontend API" — SHWIRE-01, SHLOWER-01/02, SHAPI-01 acceptance detail; out-of-scope table
- `.planning/ROADMAP.md` § "Phase 6" — the 4 success criteria (single-parse guard, isinstance relaxation, addrspace(3) IR evidence, subview GEP evidence)
- `.planning/PROJECT.md` — locked decisions, `ScalarType`↔dtype mapping, layering constraints (codegen_fns hook, no NVIDIA imports in frontend)

### Prior phase decisions
- `.planning/phases/04-c-templates-clang-ast-foundation/04-CONTEXT.md` — D-01..D-08 (SharedLinearLayout/SharedTensor design, zero-length array storage, operator() accessor, swizzle parity)
- `.planning/phases/05-mlir-op-relaxation-spec-extraction/05-CONTEXT.md` — D-09..D-11 (ODS relaxation, variant SpecInput, spec JSON schema)

### CUDA wiring (files being modified — SHWIRE-01)
- `third_party/nvidia/backend/compiler.py:195-320` — `InferExternCallResult` hook: `create_and_suspend`, `infer_result` (degenerate-basis TensorParameter construction at :276-295 — the pattern D-12 mirrors), `_scalar_type_for`
- `third_party/nvidia/backend/compiler.py:709-872` — `_pre_compile_extern_calls()`: spec JSON consumption, suspended-compiler path (:771-806), module attrs, parse-count guard (:863-871); the single-parse assertion at :676-684
- `python/src/llvm.cc:1007-1040` — existing `llvm.SharedTensorParameter` pybind binding (type/shape/offset_bases/block_bases/rank/alignment)
- `python/src/clang_compiler.h:142-165` — `SharedLayoutInfo`, `SharedTensorParameter`, `CudaFuncRequest::ParamTypes` variant

### LLVM lowering (files being modified — SHLOWER-01/02)
- `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` — the full lowering: module-attr JSON parsing pattern (:13-69), alloca+store+ptr distributed path (:141-152), call construction; the shared branch lands here
- `include/triton/Conversion/TritonGPUToLLVM/Utility.h:430,454` — `getShmemOffset`, `getSharedMemoryObjectFromStruct` declarations
- `lib/Conversion/TritonGPUToLLVM/Utility.cpp:1366-1410` — `SharedMemoryObject::getShmemOffset` (subview offset via pseudoinvert), `getShmemAffineBase` (base+offset GEP combined); canonical usage at :623
- `lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp:36,56` — `convertMemDescType` (what a memdesc operand becomes post-conversion)

### Frontend API (files being modified — SHAPI-01)
- `python/triton/experimental/gluon/language/_semantic.py:250-319` — `call_extern()`: isinstance check (:253-255), f64 guard (:259-263), arg_params construction (:270-276), result-type building
- `python/triton/experimental/gluon/language/_core.py:185,291-520` — `shared_memory_descriptor_type` / `shared_memory_descriptor` (load/store/slice/index/reshape/_reinterpret); `gl.call` at :92 (builtin)
- `python/triton/experimental/gluon/language/_layouts.py:174-213` — Python `SharedLinearLayout` / `SwizzledSharedLayout` / `NVMMASharedLayout` (the objects D-21 passes through arg_params)

### clang AST bridge (TypeBuilder addrspace change — D-15)
- `python/src/clang_compiler.h:289,380` — `TypeBuilder::BuildSharedLinearLayout` / `BuildSharedTensor` (where the LangAS::cuda_shared qualifier is applied)
- `python/src/clang_compiler.cc` — implementations; spec extraction at :1421-1560 (Phase-5 variant + JSON serialization, source of memory_space info)

### Test infrastructure
- `test/` lit config (`test/lit.cfg.py`); Phase-5 lit tests for `extern_call` mixed-operand syntax (reference for the new lowering lit test)
- `python/test/gluon/test_extern_call.py` — existing E2E patterns (Phase 7 will extend; not modified this phase)
- `AGENTS.md` § "Extern CUDA C++ Interop" — pipeline overview and file map

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`getShmemAffineBase` (`Utility.cpp:1401`)**: combines base extraction + `getShmemOffset` GEP in one helper — the shared lowering branch may reduce to a single call; canonical usage at `Utility.cpp:623`.
- **Module-attr JSON pattern (`ExternCallOpToLLVM.cpp:13-69`)**: `getMangledName`/`getExtractorNames` show exactly how to add the new `ttg.extern_call_arg_spaces` attribute read (D-16).
- **Degenerate-basis construction (`compiler.py:276-295`)**: the exact all-zero-bases pattern D-12 mirrors for `SharedTensorParameter`.
- **`llvm.SharedTensorParameter` binding (`llvm.cc:1007`)**: already exposes offset_bases/block_bases/rank/alignment — Phase 6 only constructs and routes it, no new binding needed.
- **f64 guard style (`_semantic.py:259-263` + `compiler.py:754-758`)**: the two-layer (frontend + backend) guard pattern D-20 replicates for PaddedSharedLayout.

### Established Patterns
- **Suspended-compiler single-parse flow**: `create_and_suspend` (semantic) → `infer_result` (semantic) → `compile_bitcode` (llir); shared args must flow through the same `CudaFuncRequest.param_types` variant without triggering a second parse.
- **Spec JSON → param_types loop (`compiler.py:786-796`)**: per-input dict→TensorParameter construction; the shared branch adds a `memory_space`-keyed variant arm building `SharedTensorParameter`.
- **Module attributes as Python→C++-pass side-channel**: mangled names, extractor names, and (new) arg spaces all flow via `StringAttr` JSON on the ModuleOp.

### Integration Points
- `_semantic.py:253-255` — isinstance relaxation entry point (SHAPI-01).
- `compiler.py:270-276` (`infer_result` caller side in `_semantic.py`) — arg_params dict gains `memory_space` key (D-14).
- `compiler.py:786` / `:821` — both spec-consumption loops (suspended + fallback) need the SharedSpecInput branch.
- `ExternCallOpToLLVM.cpp:141-152` — the per-operand loop where the spec-JSON-driven shared branch splits off (D-16/D-17).
- `clang_compiler.h:380` `BuildSharedTensor` — LangAS::cuda_shared qualifier application (D-15).

</code_context>

<specifics>
## Specific Ideas

- The user explicitly wants the AS3-erasure landmine (L-01) documented in this phase's artifacts so implementers and Phase-7 verifiers know to watch for `ld.generic` degradation — from firsthand experience hitting this with LLVM addrspace(3) pointers stored to memory.
- Error messages follow the existing f64 guard voice: what's unsupported, what to use instead, provenance reference.
- The lit test is a single file covering mixed shared+distributed args in one `ttg.extern_call` — validating both lowering paths AND argument ordering at once.

</specifics>

<deferred>
## Deferred Ideas

- **AS3 pointer preservation across store/reload** (MemorySSA-class analysis to keep `ld.shared` when an AS3 pointer round-trips through memory) — too complex for this phase; recorded as landmine L-01 only.
- **`PlaceholderSharedLayout` fallback** for device functions that pin a concrete `SharedLinearLayout` in their signature — not needed under the deduced-layout assumption (D-13); revisit if real users write pinned-layout functions.
- Prior deferrals remain: shared-memory return (SHRET-01), PaddedSharedLayout, dynamic `extern __shared__`, auto-barriers, AUTO-01, FP64-01.

</deferred>

---

*Phase: 6-CUDA Wiring + LLVM Lowering + Frontend API*
*Context gathered: 2026-07-16*
