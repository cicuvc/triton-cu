# Phase 4: C++ Templates + Clang AST Foundation - Context

**Gathered:** 2026-07-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the CUDA C++ device-side type layer and the clang AST bridge for shared-memory interop, standalone from any MLIR/GPU integration:

- New `SharedLinearLayout` and `SharedTensor<dtype, shape, layout>` C++ device templates in `python/test/gluon/tt_plugin.cu`.
- Extend the v1.0 clang inference infra (`clang_compiler.cc`/`.h`) with parallel `SharedLayoutInfo` / `SharedTensorParameter` structs, `TypeBuilder::BuildSharedLinearLayout()`/`BuildSharedTensor()`, a `TypeInspector` parse branch, and `FunctionResolver` support for `SharedTensor&` params.
- A `llvm.SharedTensorParameter` pybind11 binding.

**In scope:** device templates compile; AST construct → inspect round-trip; Sema resolves `__device__` functions with `SharedTensor&` params; Python binding importable/constructable.

**Explicitly NOT this phase (later v1.1 phases):** MLIR `ttg.extern_call` ODS relaxation + `extractExternCallSpecs` (Phase 5); `_pre_compile_extern_calls` wiring, `ExternCallOpToLLVM` addrspace-3 lowering, `gl.call()` frontend acceptance (Phase 6); E2E GPU + swizzle round-trip tests (Phase 7). Covers requirements **SHTYPE-01, SHTYPE-02, SHAST-01, SHAST-02, SHAST-03**.

</domain>

<decisions>
## Implementation Decisions

### SharedLinearLayout template shape
- **D-01:** Model `SharedLinearLayout` as a **flat top-level template** — `SharedLinearLayout<OffsetBases{...}, BlockBases{...}, uint32_t Align>` with **dedicated basis-carrier structs** (own `OffsetBases`/`BlockBases` NTTP carriers), matching the success-criteria syntax literally. Do NOT nest it inside a `TensorLayout`-style factory. Keep it visually/structurally distinct from the distributed `Layout`.
- **D-02:** Each offset/block basis **row is a length-`rank` vector of logical-dim coordinates** (mirrors MLIR `SharedLinearEncodingAttr`: `offset`/`block` input dims → logical tensor output dims). This guarantees round-trip fidelity with the MLIR attr and the `gluon_ir.cc` extraction (`offset_bases`/`block_bases` as `List[List[int]]`). Carrier rows are `IntTuple<RANK>`-style. (Locked as the recommended default after the follow-up question was skipped — it is the only representation that round-trips against MLIR's multi-output shared layout.)

### SharedTensor storage & accessors
- **D-03:** `SharedTensor` holds a **zero-length array** `T data[]` — a base pointer at the struct address with zero storage cost — so it lowers cleanly to `ptr addrspace(3)` in Phase 6.
- **D-04:** Accessor is an **`operator()` overload** taking logical indices: indices → `SharedLinearLayout::evaluate(...)` → offset → returns **`T&`** (mutable reference ⇒ both read and write). No separate `load`/`store` methods; no `operator[]`.
- **D-05:** Device-side parameter type is `SharedTensor<...>&` (mutable reference), consistent with the existing `Tensor` convention that becomes a raw `ptr` in LLVM IR.

### Scope of evaluate()/swizzle math
- **D-06:** Implement `SharedLinearLayout::evaluate(indices...) -> offset` **fully in this phase** — it is required by `SharedTensor::operator()`.
- **D-07:** Add a **Phase-4 bit-identical parity unit test**: `evaluate()` output must match the MLIR `LinearLayout({offsetBases, blockBases})` composition (`gluon_ir.cc:102-103`) for a non-trivial swizzled layout. This deliberately **front-loads** the swizzle-parity risk that STATE.md flags for Phases 4/7, rather than deferring correctness to Phase 7.

### Verification harness (no MLIR/GPU yet)
- **D-08:** Verify via a **Python pytest module** (GPU-free, runs under `make test-unit`) that uses the `llvm` pybind bindings: construct `llvm.SharedTensorParameter`, drive the clang inference/compile entry on a synthetic `.cu` containing a `SharedTensor&` device function, and assert (a) compilation succeeds, (b) the round-tripped `SharedTensorParameter` matches the input (scalar type, shape dims, offset/block bases, alignment), (c) `FunctionResolver` resolves without substitution failure/ambiguity, (d) the parity check from D-07. Style mirrors `test_extern_call.py` minus `@gluon.jit`/GPU.

### Locked upstream (carried forward — do not re-decide)
- `SharedTensor<T,Shape,SharedLinearLayout>` is a **separate** C++ template, NOT `Tensor` with a different layout param.
- `SharedLinearLayout` is distinct from the distributed `Layout`; carries `offset_bases` + `block_bases` + `alignment`.
- Parallel `SharedLayoutInfo` / `SharedTensorParameter` structs; `SharedTensorParameter` added as a new arm of `CudaFuncRequest::ParamTypes`.
- `SharedTensor` is argument-only for v1.1 (returning shared memory deferred to SHRET-01).

### Agent's Discretion
- **TypeInspector variant integration** (5th gray area, not discussed): planner/researcher choose how `SharedTensorParameter` joins the existing return-type inference `variant<nullptr_t, TensorParameter, TupleType>` in `DispatchTypeParsing` — new variant arm vs parallel dispatch path vs discriminated `TensorParameter`. SHAST-03 requires shared args flow through the same resolution flow; the exact variant plumbing is left to planning.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & milestone intent
- `.planning/REQUIREMENTS.md` — v1.1 requirement IDs SHTYPE-01/02, SHAST-01/02/03 with acceptance detail; out-of-scope table (PaddedSharedLayout, dynamic shared, shared return).
- `.planning/PROJECT.md` — milestone goal, locked decisions, `ScalarType`↔triton dtype string mapping, layering constraints.
- `.planning/STATE.md` — Blockers/Concerns: swizzle-parity requirement (C++ `evaluate()` must equal MLIR `LinearLayout({offsetBases,blockBases})` at `gluon_ir.cc:102-103`).

### C++ device templates (the file being extended)
- `python/test/gluon/tt_plugin.cu` — existing `Shape`, `IntTuple<N>`, `TensorLayout<Shape,N_WARPS>::Layout<REGS,LANES,WARPS>` w/ `BasisGroup<N>` + `evaluate()` (tt_plugin.cu:30-79), and `Tensor<T,Shape,Layout>` w/ `T data[REG_SIZE]` (tt_plugin.cu:85-99). Model `SharedLinearLayout`/`SharedTensor` alongside these.

### Clang AST bridge (the infra being extended)
- `python/src/clang_compiler.h` — `LayoutInfo`, `TensorParameter`, `CudaFuncRequest::ParamTypes` variant (line 167), `TypeBuilder` (BuildLayoutFactory/BuildBasisGroup/BuildLayout/BuildTensor, 224-257), `TypeInspector` (ParseLayoutType/ParseTensorType/DispatchTypeParsing, 263-280), `FunctionResolver` (LookupFunction/InstantiateFunction, 286-296).
- `python/src/clang_compiler.cc` — implementations of the above (`BuildLayout`/`BuildTensor`, `DispatchTypeParsing`, `FunctionResolver`); parallel shared path is added here.
- `python/src/llvm.cc` — pybind bindings for `ScalarType`, `TensorParameter`, `compile_cuda_to_module`, etc.; add `SharedTensorParameter` binding here.

### Reference implementation (POC to mirror)
- `/home/cicuvc/cs/project/nks/lab/cu_compiler_v2.cpp` / `.h` — standalone POC with `TypeBuilder`/`TypeInspector`/`FunctionResolver`/`EvaluateFunctionReturnType`; the return-type inference pattern the shared path parallels.

### Layout round-trip (MLIR ↔ Python ↔ clang)
- `python/src/gluon_ir.cc:225-245` — MLIR `SharedLinearEncodingAttr` → Python `SharedLinearLayout(offset_bases, block_bases, alignment)`; the parity reference for D-07.
- `python/triton/experimental/gluon/language/_layouts.py:174-213` — Python `SharedLinearLayout` / `SharedLayout` / `NVMMASharedLayout` / `SwizzledSharedLayout` definitions.
- `.planning/codebase/ARCHITECTURE.md` §"TensorParameter / clang AST Bridge" — the existing forward/reverse round-trip pattern.
- `AGENTS.md` §"Extern CUDA C++ Interop" and §"Layout Round-Trip" — pipeline overview and file map.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `IntTuple<RANK>` + `BasisGroup<N>::evaluate()` (tt_plugin.cu:16-60): the exact constexpr basis-evaluation pattern to mirror for `SharedLinearLayout::evaluate()`. Low-level primitive is reusable even though the public `SharedLinearLayout` shape is flat (D-01).
- `TypeBuilder::BuildBasisGroup`/`BuildLayout`/`BuildTensor` (clang_compiler.h:243-256): direct templates for the parallel `BuildSharedLinearLayout`/`BuildSharedTensor`.
- `TypeInspector::ParseLayoutType`/`ParseTensorType` (clang_compiler.h:270-279): parallel for shared-tensor parsing branch in `DispatchTypeParsing`.
- `LookupFunctionWithPlaceholderFallback` + `PlaceholderLayout` (clang_compiler.h:356-360, tt_plugin.cu:81-98): existing fixed-layout resolution mechanism — check whether `SharedTensor&` params need the same fallback.
- `getScalarTypeFromQualType`/`getQualTypeFromScalarType` (clang_compiler.h:182-218): scalar-type mapping reused as-is.

### Established Patterns
- Parallel-struct approach: `SharedLayoutInfo`/`SharedTensorParameter` mirror `LayoutInfo`/`TensorParameter` (SHAST-01), added to the `ParamTypes` variant rather than mutating existing structs.
- Zero-cost device storage: `Tensor` uses a fixed `T data[REG_SIZE]` value array; `SharedTensor` diverges to a zero-length `T data[]` base pointer (D-03) because it aliases external shared memory, not registers.
- pybind class exposure follows the existing `TensorParameter` binding in `llvm.cc`.

### Integration Points
- `CudaFuncRequest::ParamTypes` variant (clang_compiler.h:167) — where `SharedTensorParameter` plugs in.
- `DispatchTypeParsing` return variant (clang_compiler.h:278) — where the shared parse branch and the Agent's-Discretion integration decision (D-05 discretion note) land.
- `llvm.cc` bindings — new `SharedTensorParameter` Python class; the pytest harness (D-08) is the sole consumer this phase.

</code_context>

<specifics>
## Specific Ideas

- SharedTensor accessor is specifically `operator()` (not `operator[]`), returning `T&`, with the offset computed by `layout.evaluate(indices...)` — user's explicit design.
- SharedLinearLayout template signature literally shaped like the roadmap success criterion: `SharedLinearLayout<OffsetBases{...}, BlockBases{...}, 16>`.
- The parity unit test should exercise a **non-trivial swizzled** layout (not a trivial identity) so offset/block bases are tested independently.

</specifics>

<deferred>
## Deferred Ideas

- **Research question (not a scope change):** the D-07 parity test needs a way to invoke C++ `evaluate()` from Python — either a temporary host-callable binding, a constexpr evaluation path, or comparing offsets emitted through the existing compile path. Researcher to determine the cleanest mechanism; does not expand phase scope.
- `PaddedSharedLayout` support — out of scope for v1.1 (padding doesn't map to a shared linear layout).
- Dynamic / `extern __shared__` variable-size shared allocation — out of scope (fixed-shape only).
- Returning a `shared_memory_descriptor` from `gl.call()` (SHRET-01) — future milestone.

</deferred>

---

*Phase: 4-C++ Templates + Clang AST Foundation*
*Context gathered: 2026-07-12*
