# Project Research Summary

**Project:** triton-cu — Milestone v1.1: Shared Memory Interop for `gl.call()`
**Domain:** Compiler internals — CUDA/MLIR/Clang shared-memory interop
**Researched:** 2026-07-12
**Confidence:** HIGH

## Executive Summary

This is a **compiler-internals milestone** extending the triton-cu fork's `gl.call()` CUDA C++ interop feature. v1.1 adds shared-memory argument passing: a Gluon `shared_memory_descriptor` flows through the six-layer MLIR/LLVM/Clang pipeline and emerges as a `SharedTensor<dtype, shape, layout>&` parameter in a JIT-compiled CUDA `__device__` function, with read+write access to the underlying `addrspace(3)` shared memory. The approach extends the existing v1.0 `TensorParameter`/`TypeBuilder`/`TypeInspector`/`FunctionResolver` infrastructure with a parallel `SharedTensorParameter`/`SharedLayoutInfo` data path at every pipeline layer.

The recommended approach is a **five-phase build-up** following the natural dependency order of the compiler stack: (1) C++ device templates and clang AST extension, (2) MLIR op-signature relaxation and spec extraction, (3) CUDA compilation wiring, (4) LLVM lowering with per-operand address-space dispatch, (5) frontend integration followed by E2E verification. Phases 1 and 2 can run in parallel; Phase 4 is the highest-risk step (address-space mismatch between the caller's `alloca`-produced `ptr addrspace(0)` and the callee's expected `ptr addrspace(3)` produces silent data corruption).

**Key risk:** The `extern_call` ODS type relaxation at `TritonGPUOps.td:803` is the single change with the widest blast radius — it gates every downstream phase. The addrspace-3 lowering in `ExternCallOpToLLVM.cpp:141-152` is the top correctness risk. The v1.0 single-parse guard (`sExternCudaParseCount` asserted at `compiler.py:683`) must be preserved by routing `SharedTensorParameter` through the existing suspended-compiler path. **Full arbitrary swizzle** (`offset_bases` + `block_bases` + `alignment`) is in scope per explicit user decision; the swizzle offset-math must be bit-identical between the MLIR `SharedLinearEncodingAttr` and the C++ `SharedLinearLayout` template.

**Critical correction to prior documentation:** AGENTS.md line 15 incorrectly states that shared memory is represented as MLIR `memref`. The accurate representation is `ttg::MemDescType` (TritonGPUTypes.td:23), which lowers to `LLVM::LLVMPointerType` with address space 3 via `SharedMemoryObject` (Utility.h:374). All v1.1 work must use `MemDescType`/`SharedMemoryObject` APIs, never MLIR `MemRefType`.

## Key Findings

### Recommended Stack

The technology stack is entirely defined by the existing triton-cu codebase. v1.1 extends the v1.0 pipeline rather than introducing new external dependencies. Shared memory is already modeled in Triton's MLIR dialect as `ttg::MemDescType` with `#ttg.shared_memory` space attribute, lowered via `SharedMemoryObject` to LLVM `ptr addrspace(3)`. The target encoding is `SharedLinearEncodingAttr` (`#ttg.shared_linear`), which already exists in ODS at `TritonGPUAttrDefs.td:395-426` and has Python bindings at `_layouts.py:631-673`. On the CUDA C++ side, the existing `tt_plugin.cu` defines `Shape`, `TensorLayout`, and `Tensor` templates that serve as the model for the new `SharedLinearLayout` and `SharedTensor`. Clang's NVPTX CodeGen naturally assigns `ptr addrspace(3)` to `__device__` function parameters that reference shared memory — no special clang API is needed.

**Core technologies:**

- **LLVM 23.0.0git** (pre-23 development, self-compiled): NVPTX backend, `LLVM::LLVMPointerType::get(ctx, 3)`, `CloneFunctionInto` for bitcode linking — no version-specific breaking changes; address space 3 is stable since LLVM 15
- **Clang (in-process, via `CUDACompiler`):** CodeGen for `__device__` functions, Sema template deduction for `SharedTensor<T,Shape,SL>&`, AST construction via `TypeBuilder` — the existing coroutine-based `CUDACompiler` (`clang_compiler.cc`) is reused
- **Triton MLIR dialects (Triton/TritonGPU):** `ttg::MemDescType`, `SharedLinearEncodingAttr`, `SharedMemorySpaceAttr`, `SharedMemoryObject` utility, `toLinearLayout` conversion — all exist and need wiring, not creation
- **Python bindings (`llvm.cc`):** `TensorParameter`, `ScalarType`, `CudaFuncRequest` — extended with `SharedTensorParameter`/`SharedLayoutInfo`
- **CUDA C++ device templates (`tt_plugin.cu`):** New `SharedLinearLayout` (offset/block bases + alignment) and `SharedTensor<T,Shape,SharedLinearLayout>` — modeled on existing `Layout`/`Tensor` at `tt_plugin.cu:31-99`

### Expected Features

**Must have (table stakes) — milestone fails without these:**

- `extern_call` op accepts `MemDescType` operands (ODS relaxation at `TritonGPUOps.td:803`) — gating change for all downstream work
- `SharedLinearLayout` C++ device template in `tt_plugin.cu` — must be a **separate template** (not reuse `TensorLayout`), because shared memory has fundamentally different addressing (offset/block bases vs reg/lane/warp bases) and the `data[]` array is in addrspace 3, not registers
- `SharedTensor<T,Shape,SharedLinearLayout>&` C++ device template — the mutable-reference type the callee receives; distinct from `Tensor` (separate template, not template parameter substitution)
- `TypeBuilder::BuildSharedLinearLayout()` + `BuildSharedTensor()` — construct clang AST types from `SharedTensorParameter`; parallel to existing `BuildLayout()`/`BuildTensor()` at `clang_compiler.cc:740-767`
- `TypeInspector` shared-tensor parsing (`ParseSharedTensorType`, `ParseSharedLayoutType`) — reverse clang AST → `SharedTensorParameter` for type-inference compatibility; extends `DispatchTypeParsing()` at `clang_compiler.h:278`
- `extractExternCallSpecs()` handles `MemDescType` — the `cast<RankedTensorType>` at `clang_compiler.cc:1170` must become a `dyn_cast` branch; shared-memory specs get `"memory_space": "shared"` + `offset_bases`/`block_bases`/`alignment` JSON keys
- Full arbitrary swizzle support (`offset_bases` + `block_bases` + `alignment`) — the `SharedLinearLayout` template must compute byte offsets bit-identical to the MLIR `LinearLayout({offsetBases, blockBases}, outDims)` composition at `gluon_ir.cc:102-103`; this includes arbitrary basis matrices, not just strided patterns
- Shared-memory arg lowering via `SharedMemoryObject::getBase()` pass-through — the existing `alloca+store+ptr` loop at `ExternCallOpToLLVM.cpp:143-152` must dispatch per-operand; memdesc operands extract `SharedMemoryObject` and pass `getBase()` (ptr addrspace 3) directly, NOT alloca/store (which produces addrspace 0)
- Callee LLVM signature verification — confirm clang's natural `ptr addrspace(3)` emission for `__device__ SharedTensor&` params matches the caller's `smemObj.getBase()` type
- Frontend `call_extern()` allows `shared_memory_descriptor` args — the `isinstance(a, ttgl.tensor)` check at `_semantic.py:254` must also accept `ttgl.shared_memory_descriptor`
- Subview offset accumulation preserved — `getSharedMemoryObjectFromStruct()` at `Utility.cpp:1428-1452` must be used (not raw `getSharedMemoryBase()`) to include offsets from `memdesc_subslice`/`memdesc_index`/`memdesc_reshape`
- Single-parse guard preserved — `SharedTensorParameter` must flow through the suspended-compiler path at `compiler.py:776`; the parse-counter assertion at `compiler.py:683` must hold
- E2E test: shared-memory read+write through `gl.call()` with `gl.barrier()` synchronization

**Should have (differentiators — built within this milestone):**

- Mutable shared-memory parameter (`SharedTensor&`, not `const SharedTensor&`) — enables in-place reductions, scratch space, atomics through `gl.call()`; `mutableMemory=true` is already set at `gluon_ir.cc:315`
- Integration with v1.0 return-type inference — `TypeInspector::DispatchTypeParsing()` recognizes `SharedTensor` but does not yet use it for result inference (v1.1 is argument-only); the infrastructure is laid for future shared-memory return types
- Reuse of existing `ModuleAllocation` + `AllocateSharedMemory` pass — no new allocation analysis needed; shared memory for `gl.call()` args uses the same `allocation.offset` attributes as all other shared-memory ops at `AllocateSharedMemory.cpp:17-26`

**Defer (v2+):**

- Returning a `shared_memory_descriptor` from `gl.call()` — requires shared-memory liveness tracking across the caller/callee boundary (decision 2026-07-12, PROJECT.md:63)
- Auto-deriving `result_layout` from CUDA-inferred layout (AUTO-01) — deferred per decision 2026-07-11
- Dynamic shared memory allocation (`extern __shared__`) from `gl.call()` — would require CUDA dynamic-shmem ABI support; use only fixed-shape allocs
- Frontend syntactic sugar (auto-detecting shared-memory args) — simple validation change, done last in the integration phase

### Architecture Approach

The v1.1 change extends the existing six-layer `gl.call()` pipeline at a well-defined seam in each layer, rather than building parallel infrastructure. Each layer gets one focused extension:

1. **Python Gluon Frontend** (`_core.py`, `_semantic.py`): Allow `shared_memory_descriptor` args alongside tensors; attach `is_shared` + layout info to `arg_params` for the inference hook
2. **MLIR Op Layer** (`TritonGPUOps.td:803`): Relax `Variadic<TT_Tensor>:$inputs` to accept `MemDescType` — this is the single most impactful change (widest blast radius)
3. **Spec Extraction** (`clang_compiler.cc:1169`): Branch on `isa<MemDescType>` in `extractExternCallSpecs()`; extract `toLinearLayout()` bases into JSON `SpecInput` with `"memory_space": "shared"` key
4. **CUDA JIT Compilation** (`compiler.py:771`, `clang_compiler.h`, `clang_compiler.cc`): New `SharedTensorParameter`/`SharedLayoutInfo` structs; `TypeBuilder::BuildSharedTensor()` constructs clang AST; `FunctionResolver::LookupFunction()` handles `SharedTensor&` args via existing Sema template deduction; all flows through the suspended-compiler path to preserve the single-parse guard
5. **LLVM Lowering** (`ExternCallOpToLLVM.cpp:141`): Per-operand dispatch — memdesc operands extract `SharedMemoryObject` via `getSharedMemoryObjectFromStruct()`, apply accumulated subview offsets via `getShmemOffset()`, pass `ptr addrspace(3)` directly; tensor operands keep existing `alloca+store+ptr` path
6. **Bitcode Linking** (`clang_compiler.cc`, `compiler.py:387`): No change — `linkBitcodeToModule()` is type-agnostic; callee's `ptr addrspace(3)` signature is preserved through linking

**New cross-boundary data structures** (all in `clang_compiler.h`):
- `SharedLayoutInfo { vector<uint32_t> OffsetBases, BlockBases; uint32_t Alignment }` — alongside existing `LayoutInfo`
- `SharedTensorParameter { ScalarType Type; vector<uint32_t> Shape; SharedLayoutInfo Layout }` — alongside `TensorParameter`
- `CudaFuncRequest::ParamTypes` extended with `SharedTensorParameter` variant

**Key design decision — separate `SharedTensor` template:** The `SharedTensor<T,Shape,SharedLinearLayout>` template must be a separate C++ template (not `Tensor<T,Shape,SharedLinearLayout>`), because: (a) `SharedTensor`'s `data[]` array lives in shared memory (addrspace 3), not registers; (b) `SharedLinearLayout` has fundamentally different `evaluate()` semantics (offset+block bases vs reg+lane+warp bases); (c) a separate template enables compile-time dispatch on `__shared__` storage class.

### Critical Pitfalls

1. **Address-space mismatch (PIT-1):** The existing lowering at `ExternCallOpToLLVM.cpp:143-152` applies `alloca+store+ptr` (produces `ptr addrspace(0)`) to every operand. Shared-memory callees expect `ptr addrspace(3)`. Passing addrspace 0 where addrspace 3 is expected causes silent data corruption — reads/writes go to generic memory instead of shared memory. **Prevent by:** per-operand dispatch — for memdesc operands, extract `SharedMemoryObject` via `getSharedMemoryObjectFromStruct()` and pass `getBase()` (already `ptr addrspace(3)`) directly; no alloca/store. Must also apply accumulated subview offsets via `getShmemOffset()`.

2. **`cast<RankedTensorType>` crash on `MemDescType` (PIT-2):** `extractExternCallSpecs()` at `clang_compiler.cc:1170` unconditionally casts every operand to `RankedTensorType`. A `MemDescType` operand triggers an LLVM `cast<>` assertion failure. **Prevent by:** replacing the `cast<>` with a `dyn_cast`/`isa` branch — `MemDescType` operands extract shared layout via `toLinearLayout(memDescType)` at `LinearLayoutConversions.cpp:1376`.

3. **ODS `Variadic<TT_Tensor>` rejects `MemDescType` (PIT-3):** The `extern_call` op at `TritonGPUOps.td:803` is constrained to `Variadic<TT_Tensor>:$inputs`. `MemDescType` does not satisfy `TT_Tensor`. MLIR verification fails at op creation. **Prevent by:** relaxing to `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs`. This is the single change with the widest blast radius — every consumer of `extern_call` ops (verifier, spec extractor, lowering pattern, IR printer/parser) must handle both types.

4. **Swizzle/offset-bases mismatch between MLIR and CUDA C++ (PIT-4):** The MLIR `LinearLayout({offsetBases, blockBases}, outDims)` composition at `gluon_ir.cc:102-103` computes byte offsets. The C++ `SharedLinearLayout` template must compute **identical** byte offsets. A mismatch means wrong shared-memory addresses — silently wrong results. **Prevent by:** a single-source-of-truth formula in both sides; E2E test that writes through `SharedTensor&` at specific logical positions and reads back via `shared_memory_descriptor.load()`, verifying bit-for-bit match for each basis independently.

5. **Double-parse breaking the single-parse guard (PIT-8):** The v1.0 infrastructure asserts `parse_count_delta == distinct_cu` at `compiler.py:683`. If v1.1 shared-memory spec extraction creates a new `CUDACompiler` instead of reusing the suspended one, the parse counter increments and the assertion fires. **Prevent by:** routing `SharedTensorParameter` through the existing suspended-compiler path at `compiler.py:776`; extend `CudaFuncRequest::ParamTypes` with `SharedTensorParameter` variant.

6. **Subview offset accumulation not preserved (PIT-5):** When a user slices/indexes/reshapes shared memory before passing to `gl.call()`, the `SharedMemoryObject.offsets` accumulate subview offsets. Passing `getBase()` without `getShmemOffset()` gives the allocation start, not the subview start — the callee operates on the wrong memory. **Prevent by:** computing `base + shmemOffset` via GEP before passing to the callee; use `getSharedMemoryObjectFromStruct()` (not raw `getSharedMemoryBase()`).

7. **Read+write visibility without synchronization (PIT-12):** `gl.call()` with `SharedTensor&` allows the callee to write shared memory. After the call returns, other Gluon code may read it — but no `__syncthreads()` barrier is inserted automatically. This is a race condition at the warp/CTA level. **Prevent by:** explicit `gl.barrier()` calls around shared-memory-accessing `gl.call()` invocations; document this requirement clearly; E2E test with barriers for deterministic results.

## Implications for Roadmap

Based on combined research, the recommended phase structure follows the dependency order of the compiler stack. Phases 1 and 2 are independent and can be parallelized. Each phase must address specific pitfalls and deliverables.

### Phase 1: C++ Templates + Clang AST Extension (Foundation)

**Rationale:** The C++ `SharedTensor` and `SharedLinearLayout` types are the semantic foundation every other layer references. The clang AST bridge must construct and parse these types before any MLIR or lowering work can proceed. The `CudaFuncRequest::ParamTypes` variant must be extended here to preserve the single-parse guard.

**Delivers:**
- `SharedLinearLayout` template in `tt_plugin.cu` (offset/block bases + alignment; full swizzle)
- `SharedTensor<T,Shape,SharedLinearLayout>` template in `tt_plugin.cu` (separate from `Tensor`)
- `SharedLayoutInfo`, `SharedTensorParameter` structs in `clang_compiler.h`
- `CudaFuncRequest::ParamTypes` extended with `SharedTensorParameter` variant
- `TypeBuilder::BuildSharedLinearLayout()` and `BuildSharedTensor()` methods in `clang_compiler.cc`
- `TypeInspector::ParseSharedLayoutType()` and `ParseSharedTensorType()` methods; `DispatchTypeParsing()` branch
- Python bindings: `llvm.SharedTensorParameter()` class exposed from `llvm.cc`

**Avoids:** PIT-8 (double-parse) — extends `CudaFuncRequest` before Phase 3 wiring; PIT-4 (swizzle mismatch) — C++ template must match MLIR `LinearLayout` composition from the start

**Research flag:** LOW — the existing `Layout`/`Tensor` templates provide a well-practiced pattern; `TypeBuilder`/`TypeInspector` extension follows established architecture

### Phase 2: MLIR ODS Relaxation + Spec Extraction

**Rationale:** The `extern_call` op must accept memdesc operands before any integration test can pass. The spec extraction must handle mixed types to connect MLIR to CUDA compilation. This is the single change with the widest blast radius — it gates every downstream phase.

**Delivers:**
- `TritonGPUOps.td:803` — relax `Variadic<TT_Tensor>:$inputs` to `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs`
- `clang_compiler.cc:1169-1213` — `extractExternCallSpecs()` `dyn_cast` branch for `MemDescType`
- `SpecInput` extended with `isShared`, `offsetBases`, `blockBases`, `alignment`
- JSON serialization extended with `"memory_space": "shared"` + `"offset_bases"`/`"block_bases"`/`"alignment"` keys
- **Lit test:** `extern_call` with mixed tensor+memdesc inputs passes verification

**Avoids:** PIT-3 (ODS rejection), PIT-2 (`cast<RankedTensorType>` crash), PIT-9 (MLIR verification failure post-relaxation — must handle `SameVariadicOperandSize` trait)

**Research flag:** MEDIUM — `AnyTypeOf` with mixed variadic types requires careful verification that all downstream passes handle the relaxed constraint; the ODS change has a wide blast radius across the MLIR module

### Phase 3: CUDA Compilation Wiring (Python ↔ C++)

**Rationale:** Connect the MLIR-side JSON specs to the CUDA compiler so shared-memory args flow through the suspended-compiler path. Without this, specs are extracted but never consumed.

**Delivers:**
- `compiler.py:771-799` — `_pre_compile_extern_calls()` builds `SharedTensorParameter` for shared inputs
- `compiler.py:820-830` — same for fallback path
- `llvm.cc` — `SharedTensorParameter` pybind11 class with `.type`, `.shape`, `.offset_bases`, `.block_bases`, `.alignment`
- `_pre_compile_extern_calls()` dispatches `llvm.SharedTensorParameter` to requests via `CudaFuncRequest`
- Verify the suspended-compiler path at `compiler.py:776` carries `SharedTensorParameter`

**Avoids:** PIT-8 (double-parse) — ensures `SharedTensorParameter` uses the suspended compiler, not a fresh parse; the assertion at `compiler.py:683` must hold

**Research flag:** LOW — pure wiring; follows the existing `TensorParameter` pattern exactly

### Phase 4: LLVM Lowering — Shared Memory Arg Pass-Through (Highest Risk)

**Rationale:** The lowering must pass `ptr addrspace(3)` directly to the callee, applying accumulated subview offsets. Without this, the call site and callee signatures mismatch, producing silent data corruption. This is the top correctness risk in the pipeline.

**Delivers:**
- `ExternCallOpToLLVM.cpp:141-152` — per-operand dispatch: `isa<MemDescType>` branch extracts `SharedMemoryObject` via `getSharedMemoryObjectFromStruct()`, applies `getShmemOffset()` via GEP, passes `ptr addrspace(3)` directly; tensor branch keeps existing alloca+store path
- Tuple-return path verified with mixed args (L171-233 already works with `promotedTypes`)
- **LLVM IR inspection test:** caller's `call` instruction passes `ptr addrspace(3)` for shared-memory positions, matching callee's `ptr addrspace(3)` parameter

**Avoids:** PIT-1 (address-space mismatch), PIT-5 (subview offset accumulation), PIT-7 (mixed arg ordering — the lowering processes operands in MLIR order, which must match the C++ function's declared parameter order)

**Research flag:** HIGH — the `ExternCallOpToLLVM` lowering is the most correctness-sensitive change; address-space mismatch produces silent data corruption, not crashes; must include LLVM IR dump inspection in verification

### Phase 5: Frontend Integration + E2E Verification

**Rationale:** The `gl.call()` API must accept `shared_memory_descriptor` arguments and the full pipeline must pass end-to-end tests. This phase gates the milestone's completion.

**Delivers:**
- `_core.py:803` — `_semantic.to_tensor(a)` adjusted for memdesc pass-through
- `_semantic.py:254` — `isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor))` check
- `_semantic.py:270-278` — `arg_params` extended with `"is_shared": True` + shared layout info
- New E2E test in `python/test/gluon/test_extern_call.py`: shared-memory read+write through `gl.call()` with `gl.barrier()` synchronization
- Swizzle-correctness test: full arbitrary swizzle `SharedLinearLayout`, write through `SharedTensor&` at specific indices, read back via `shared_memory_descriptor.load()`, verify bit-for-bit match
- Subview offset test: `smem.slice()` → `gl.call()` → verify writes land in the correct subview
- Regression: all 6 existing `test_extern_call.py` tests pass unchanged
- Documentation: `gl.call()` API docs warn about `gl.barrier()` requirement around shared-memory-accessing calls

**Avoids:** PIT-4 (swizzle mismatch — E2E test with specific index checking), PIT-6 (alignment violation — test with non-trivial swizzle bases), PIT-11 (ret-type fix corruption — test void-returning callee with `SharedTensor&`), PIT-12 (read+write hazard — test with explicit barriers; document the requirement), PIT-10 (frontend validation rejection)

**Research flag:** MEDIUM — E2E correctness depends on all prior phases; GPU test stability requires careful barrier placement

### Phase Ordering Rationale

```
Phase 1 (C++ templates + clang AST) ────┐
                                         ├──→ Phase 3 (CUDA wiring) ──→ Phase 4 (lowering) ──→ Phase 5 (E2E)
Phase 2 (ODS relaxation + specs)  ──────┘
```

- **Phases 1 and 2 are independent** — C++ template work and MLIR op changes touch different file trees with no shared dependencies. They can be developed and reviewed in parallel.
- **Phase 3 requires both 1 and 2** — the `SharedTensorParameter` struct (Phase 1) and the JSON specs with `"memory_space": "shared"` (Phase 2) converge in `_pre_compile_extern_calls()`.
- **Phase 4 depends on Phase 3** — the lowering needs the callee's LLVM bitcode (with correct `SharedTensor&` signature) produced by the CUDA compilation wired in Phase 3.
- **Phase 5 gates on all prior phases** — E2E tests exercise the full pipeline.

**Highest-risk seam:** Phase 2's ODS relaxation at `TritonGPUOps.td:803` is the critical-path gate. If the `AnyTypeOf` constraint breaks downstream MLIR verification, every subsequent phase stalls until fixed. Mitigate by adding lit tests for `extern_call` verification with both tensor-only AND mixed-operand IR BEFORE any lowering change.

**Highest correctness risk:** Phase 4's address-space lowering. The `alloca+store+ptr` pattern at `ExternCallOpToLLVM.cpp:143-152` has been tested for tensor args but silently corrupts shared-memory args. Must add LLVM IR dump inspection as a verification step.

### Research Flags

**Phases likely needing deeper research during planning (`--research-phase`):**

- **Phase 2 (ODS relaxation):** MEDIUM — `AnyTypeOf<[TT_Tensor, TTG_MemDescType]>` with mixed variadic inputs is not a common MLIR pattern; needs verification that `SameVariadicOperandSize` trait, verifier, and all downstream consumers handle it correctly. Consider Option B (separate `$shared_inputs` variadic) if `AnyTypeOf` proves problematic.
- **Phase 4 (LLVM lowering):** HIGH — the `ExternCallOpToLLVM` change is correctness-critical; address-space mismatch is silent. The lowering must be verified with LLVM IR dump inspection, not just runtime correctness. Subview offset accumulation (`getShmemOffset()`) needs careful GEP construction.
- **Phase 5 (Swizzle correctness):** MEDIUM — the swizzle offset-math parity between MLIR `LinearLayout` and C++ `SharedLinearLayout` cannot be verified by the compiler; it requires E2E GPU testing with specific logical index patterns. A dedicated swizzle-correctness test exercising each basis independently is essential.

**Phases with standard patterns (skip research-phase):**

- **Phase 1 (C++ templates):** LOW — the `Layout`/`Tensor` template pattern at `tt_plugin.cu:31-99` is well-established; `TypeBuilder`/`TypeInspector` extension follows the documented architecture at `clang_compiler.h:224-296`.
- **Phase 3 (CUDA wiring):** LOW — pure wiring; follows the existing `TensorParameter` → `CudaFuncRequest` pattern at `compiler.py:786-795` exactly.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Every technology claim is grounded in `file:line` citations from the actual codebase. MLIR types (`MemDescType` at TritonGPUTypes.td:23, `SharedLinearEncodingAttr` at TritonGPUAttrDefs.td:395), LLVM APIs (`getSharedMemoryBase` at Utility.cpp:1569, `getSharedAddressSpace()` at TargetInfo.cpp:620), and clang integration (`TypeBuilder` at clang_compiler.h:224, `TypeInspector` at clang_compiler.h:263) are all verified. No external documentation dependency. |
| Features | HIGH | Feature requirements are derived from PROJECT.md (user decisions + scope boundary) and validated against the codebase's existing capability at every pipeline layer. The deferred features (shared-memory return, auto-derive `result_layout`) are explicitly recorded with decision dates. Full swizzle scope confirmed by user decision 2026-07-12. |
| Architecture | HIGH | The six-layer pipeline extension is mapped at every layer with `file:line` precision. The component responsibility matrix (ARCHITECTURE.md Section 6) identifies exactly which files change and by how much. The anti-patterns (ARCHITECTURE.md Section 5) prevent six known design mistakes. |
| Pitfalls | HIGH | All 12 pitfalls are grounded in specific code locations and describe concrete failure modes with reproduction steps and detection methods. The pitfall-to-phase mapping (PITFALLS.md Section 10) ties each risk to the phase that must prevent it. The "Looks Done But Isn't" checklist provides verification criteria. |

**Overall confidence: HIGH** — every claim is backed by specific file:line references in the actual codebase. The four research files independently converged on the same phase structure and the same top risks. No speculative or externally-referenced claims.

### Gaps to Address

- **`SameVariadicOperandSize` trait interaction (Phase 2):** The `extern_call` op at `TritonGPUOps.td:787` may have `SameVariadicOperandSize` constraining operand vs. result variadic sizes. If inputs become mixed-type but results remain `TT_Tensor`-only, the trait may need adjustment. **Handle during Phase 2 planning:** review the trait's semantics with `AnyTypeOf` variadic; test with `ModuleOp::verify()` after op construction.
- **Clang template `ClassTemplateSpecializationDecl` lookup for `SharedTensor` (Phase 1):** The `TypeBuilder::BuildSharedTensor()` needs to look up the `SharedTensor` template declaration in the parsed translation unit. If the template definition uses different syntax than `Tensor`, the lookup may need adjustment. **Handle during Phase 1 implementation:** verify by constructing a `SharedTensor` AST node and dumping it; the existing `llvm::StringRef("Tensor")` lookup pattern at clang_compiler.cc:740 generalizes.
- **`toLinearLayout` for `MemDescType` with non-`SharedLinearEncodingAttr` (Phase 2):** The spec extraction at `clang_compiler.cc:1169` calls `toLinearLayout(memDescType)` which works for all shared encodings. But only `SharedLinearEncodingAttr` carries explicit offset/block bases that map cleanly to the C++ `SharedLinearLayout`. If a `SwizzledSharedEncoding` or `NVMMASharedEncoding` is encountered, the bases must be materialized from `toLinearLayout()` — verify the output format. **Handle during Phase 2 planning:** restrict initial support to `SharedLinearEncodingAttr`; add a diagnostic for unsupported encodings.
- **C++ `SharedLinearLayout` shape inference parity with `_layouts.py:660-666` (Phase 5):** The Python `SharedLinearLayout` derives logical dimensions from `max_stride` via `1 << s.bit_length()`. The C++ template must replicate this exactly. **Handle during Phase 5 verification:** add a C++ helper test that computes shapes from the same bases and compares to Python output.

## Sources

### Primary (HIGH confidence — codebase file:line verified)

- `TritonGPUTypes.td:23-84` — `MemDescType` definition (Shape, elementType, encoding, memorySpace, mutableMemory, allocShape)
- `TritonGPUAttrDefs.td:395-426` — `SharedLinearEncodingAttr` ODS (LinearLayout + alignment)
- `TritonGPUAttrDefs.td:1496-1501` — `SharedMemorySpaceAttr` → `#ttg.shared_memory`
- `LinearLayoutConversions.cpp:1376-1392` — `toLinearLayout(MemDescType)`
- `gluon_ir.cc:308-318` — `get_shared_mem_desc_ty` (MemDescType builder)
- `gluon_ir.cc:454-466` — `get_shared_linear_layout` (SharedLinearEncodingAttr builder)
- `gluon_ir.cc:615-624` — `create_extern_call`
- `Utility.h:374-448` — `SharedMemoryObject` class (bases + offsets)
- `Utility.h:454-457` — `getSharedMemoryObjectFromStruct()` declaration
- `Utility.cpp:1428-1452` — `getSharedMemoryObjectFromStruct()` implementation
- `Utility.cpp:1569-1589` — `getSharedMemoryBase()` → `ptr addrspace(3)`
- `TritonNVIDIAGPUToLLVM/TargetInfo.cpp:620` — `getSharedAddressSpace() returns 3`
- `AllocateSharedMemory.cpp:17-26` — `ModuleAllocation` analysis
- `TritonGPUOps.td:786-814` — `ExternCallOp` ODS (`Variadic<TT_Tensor>:$inputs`)
- `ExternCallOpToLLVM.cpp:111-293` — ExternCallOpConversion lowering
- `clang_compiler.cc:1152-1219` — `extractExternCallSpecs()`
- `clang_compiler.cc:730-768` — `CUDACompiler::BuildTensor()` (pattern for `BuildSharedTensor`)
- `clang_compiler.cc:1340-1396` — `linkBitcodeToModule()` ret-type fix
- `clang_compiler.h:129-141` — `LayoutInfo`, `TensorParameter`
- `clang_compiler.h:165-169` — `CudaFuncRequest`
- `clang_compiler.h:224-296` — `TypeBuilder`, `TypeInspector`, `FunctionResolver`
- `compiler.py:683-686` — parse-counter assertion
- `compiler.py:190-218` — `InferExternCallResult.create_and_suspend()`
- `compiler.py:709-849` — `_pre_compile_extern_calls()`
- `tt_plugin.cu:1-185` — Existing C++ templates (Shape, TensorLayout, Tensor, PlaceholderLayout)
- `_core.py:775-809` — `gl.call()` frontend
- `_core.py:446-475` — `shared_memory_descriptor.slice()` / `.index()`
- `_semantic.py:250-318` — `call_extern()` arg validation + arg_params
- `_layouts.py:631-673` — `SharedLinearLayout` Python dataclass
- `Dialect.cpp:2061-2070` — `SharedLinearEncodingAttr::toLinearLayout()`
- `gluon_ir.cc:235-242` — `layoutToGluon` SharedLinearLayout branch
- `/media/cicuvc/.../LLVMConfigVersion.cmake` — LLVM 23.0.0git version
- `.planning/PROJECT.md` — milestone scope, key decisions, deferred features

### Secondary (MEDIUM confidence — research synthesis, not direct code reference)

- `.planning/research/STACK.md` — comprehensive type system and API reference for shared memory (373 lines, 100% codebase-cited)
- `.planning/research/FEATURES.md` — feature landscape, dependencies, and MVP prioritization (97 lines)
- `.planning/research/ARCHITECTURE.md` — six-layer pipeline extension with component responsibility matrix (656 lines)
- `.planning/research/PITFALLS.md` — 12 pitfalls with file:line roots and detection methods (461 lines)
- `AGENTS.md` — build instructions, compiler pipeline documentation (partially superseded: line 15 `memref` claim corrected by STACK.md Section 1)

---

*Research completed: 2026-07-12*
*Ready for roadmap: yes*
*Research files: STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md*
