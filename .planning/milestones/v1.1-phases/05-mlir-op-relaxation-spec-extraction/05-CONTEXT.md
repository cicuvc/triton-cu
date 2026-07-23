# Phase 5: MLIR Op Relaxation + Spec Extraction - Context

**Gathered:** 2026-07-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Relax the `ttg.extern_call` MLIR op ODS to accept `MemDescType` (shared memory) operands alongside `RankedTensorType` (register) operands, and extend `extractExternCallSpecs()` in the clang compiler to emit shared-layout specs for memdesc operands without crashing.

**In scope:** ODS `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>` relaxation; variant-based `SpecInput` to carry both distributed and shared layout data; `extractExternCallSpecs()` branching on operand type; JSON output with `memory_space`, `offset_bases`, `block_bases`, `alignment` for shared inputs; two lit tests (mixed parse + tensor-only regression).

**Explicitly NOT this phase:** `_pre_compile_extern_calls()` wiring of shared args through CUDA compilation (Phase 6); `ExternCallOpToLLVM` addrspace-3 lowering (Phase 6); `gl.call()` frontend acceptance of `shared_memory_descriptor` (Phase 6); E2E GPU + swizzle round-trip tests (Phase 7). Covers requirements **SHMLIR-01** and **SHMLIR-02**.

</domain>

<decisions>
## Implementation Decisions

### ODS relaxation (SHMLIR-01)
- **D-09:** Use `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>` — single mixed input list with per-operand type branching. Chosen over separate `Variadic<TT_Tensor>` + `Variadic<TTG_MemDescType>` argument categories because a single list preserves natural arg ordering and is MLIR-idiomatic. Downstream passes that iterate operands via `dyn_cast`/`isa` will handle mixed types correctly; any pass doing a blind `cast<RankedTensorType>` is already fragile and should be discovered/verified.

### SpecInput data model (SHMLIR-02)
- **D-10:** Replace the single `SpecInput` struct with a `std::variant<TensorSpecInput, SharedSpecInput>` — clean type-level separation. Chosen over adding `std::optional<>` fields to a single struct because sharing distributed-layout fields (`regBases`, `laneBases`, `warpBases`) with shared inputs is misleading and makes the JSON serialization ambiguous. Both variants remain within the anonymous namespace in `clang_compiler.cc`. The `ExternCallSpec::inputs` vector becomes `SmallVector<std::variant<TensorSpecInput, SharedSpecInput>, 4>`.

  - `TensorSpecInput`: existing fields — `dtype`, `shape`, `numWarps`, `regBases`, `laneBases`, `warpBases`
  - `SharedSpecInput`: new fields — `dtype`, `shape`, `memory_space` ("shared"), `offset_bases`, `block_bases`, `alignment`

### Lit test design
- **D-11:** Two lit tests, no extraction smoke test in this phase:
  1. Mixed tensor+memdesc: parse a `ttg.extern_call` with both operand types → `--verify-diagnostics` passes
  2. Tensor-only regression: existing extern_call format parses unchanged → `--verify-diagnostics` passes
  Extraction smoke testing (JSON output verification) is deferred — it requires the full `toLinearLayout` pipeline wired for memdesc, which is better tested in Phase 6 when the CUDA compiler consumes the specs.

### Locked upstream (carried forward from Phase 4 — do not re-decide)
- `SharedLinearLayout` is distinct from distributed `Layout`; carries `offset_bases` + `block_bases` + `alignment` (D-05)
- `SharedTensor` is argument-only for v1.1 — no shared memory return (D-03 scope)
- No new dialect op — relax the existing `ttg.extern_call` (locked per REQUIREMENTS.md)
- `toLinearLayout` already handles shared encodings (`SharedLinearEncodingAttr`, `SwizzledSharedEncoding`, `NVMMASharedEncoding`) at `gluon_ir.cc:1069`

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & milestone intent
- `.planning/REQUIREMENTS.md` § "MLIR Op & Spec Extraction" — SHMLIR-01 and SHMLIR-02 with acceptance detail; out-of-scope table (PaddedSharedLayout, dynamic shared, shared return)
- `.planning/PROJECT.md` — milestone goal, locked decisions, `ScalarType`↔triton dtype mapping, layering constraints; Key Decisions table
- `.planning/STATE.md` — current focus: Phase 5; Blockers/Concerns: ODS `Variadic` blast radius for downsteam passes that assume tensor-only inputs

### Prior phase decisions
- `.planning/phases/04-c-templates-clang-ast-foundation/04-CONTEXT.md` — D-05 (`SharedLinearLayout` distinct from distributed `Layout`), D-01/D-02 layout shape decisions, D-03 (`SharedTensor` zero-length array storage), locked upstream decisions

### ODS definition (the file being modified)
- `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:786-814` — `TTG_ExternCallOp` definition with `Variadic<TT_Tensor>:$inputs` (to be relaxed to `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>`)
- `include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:23-46` — `TTG_MemDescType` ODS definition

### Referenced ODS patterns to follow
- `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td` — uses `Variadic<TTG_MemDescType>` and `AnyTypeOf<[TTG_MemDescType, TT_Tensor]>` in several ops (e.g., NVWS ops:56, TritonNvidiaGPU type interfaces:84-87)
- `include/triton/Dialect/TritonInstrument/IR/TritonInstrumentOps.td:61` — `MemDescType` operand pattern

### Spec extraction (the function being modified)
- `python/src/clang_compiler.cc:1421-1505` — `SpecInput` struct, `ExternCallSpec`, `extractExternCallSpecs()` — the function that does `cast<RankedTensorType>` at line 1456 (to be changed to `dyn_cast` with memdesc branch)
- `python/src/clang_compiler.cc:1509-1560` — `tritonExtractExternCallSpecs()` JSON serialization (to be extended for shared fields)

### Shared layout round-trip
- `python/src/gluon_ir.cc:225-245` — MLIR `SharedLinearEncodingAttr` → Python shared layout; `gluon_ir.cc:1069` — `toLinearLayout` for shared encodings
- `python/triton/experimental/gluon/language/_layouts.py:174-213` — Python `SharedLinearLayout` / `SharedLayout` / `NVMMASharedLayout` / `SwizzledSharedLayout`
- `AGENTS.md` § "Extern CUDA C++ Interop" and § "Layout Round-Trip (MLIR ↔ clang AST)" — pipeline overview

### Test infrastructure
- Existing lit tests under `test/` — lit configured in `test/lit.cfg.py` and `test/lit.site.cfg.py.in`
- Existing MLIR test patterns — Gluon lit tests under `test/TritonGPU/` for reference on `extern_call` syntax

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`Variadic<AnyTypeOf<>>` ODS pattern**: already used in `TritonNvidiaGPUOps.td` and `TritonInstrumentOps.td` for ops accepting both `MemDescType` and other types — directly applicable to the SHMLIR-01 ODS relaxation.
- **`toLinearLayout` for shared encodings** (`gluon_ir.cc:1069`): already wired for `SharedLinearEncodingAttr`, `SwizzledSharedEncoding`, `NVMMASharedEncoding` — ready to call from `extractExternCallSpecs()` for memdesc operands.
- **`MemDescType` type system**: defined in `TritonGPUTypes.td:23`, has `getShape()`, `getElementType()`, `getEncoding()`, `getMemorySpace()` — same getter interface as `RankedTensorType`, simplifying the extraction branch.
- **Existing lit test patterns**: Gluon lit tests under `test/TritonGPU/` show `extern_call` syntax with encoding attributes, providing templates for the new tests.

### Established Patterns
- **`SpecInput` struct in anonymous namespace** (`clang_compiler.cc:1421`): followed by `ExternCallSpec` (line 1430), consumed by `extractExternCallSpecs()` (line 1438), serialized to JSON by `tritonExtractExternCallSpecs()` (line 1514). The variant-based replacement (D-10) preserves this pattern — new variant types go in the same anonymous namespace.
- **Dtype string mapping** (clang_compiler.cc:1482-1497): `isa<Float32Type>` → `"f32"` etc. — reused as-is for the shared branch.
- **Basis flattening** (clang_compiler.cc:1467-1472): `flattenBases()` lambda converts `LinearLayout` basis rows to flat `SmallVector<int32_t>` — applicable to `offset_bases`/`block_bases` for shared layouts.

### Integration Points
- **`TritonGPUOps.td:803`**: the `Variadic<TT_Tensor>:$inputs` constraint — replace with `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>`.
- **`clang_compiler.cc:1456`**: the `cast<RankedTensorType>` — becomes `dyn_cast` with a `MemDescType` branch that calls `toLinearLayout(shape, encoding)` and extracts `offset_bases`/`block_bases`/`alignment`.
- **`clang_compiler.cc:1519`**: JSON serialization — needs `std::visit` over the variant to emit either distributed-layout fields or shared-layout fields.

</code_context>

<specifics>
## Specific Ideas

- Variant-based replacement of `SpecInput` is specifically `std::variant<TensorSpecInput, SharedSpecInput>` — two arms, one per memory space.
- `SharedSpecInput` carries: `dtype`, `shape`, `memory_space` (always `"shared"`), `offset_bases` (flattened `SmallVector<int32_t, 16>`), `block_bases` (same type), `alignment` (`int32_t`).
- Lit test mixed input should use a `SharedLinearEncodingAttr`-encoded `MemDescType` (simplest shared encoding) alongside a `BlockedEncodingAttr`-encoded tensor.
- The tensor-only regression lit test is effectively unchanged from the existing extern_call syntax — just confirms the ODS change doesn't break existing format.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 5-MLIR Op Relaxation + Spec Extraction*
*Context gathered: 2026-07-15*
