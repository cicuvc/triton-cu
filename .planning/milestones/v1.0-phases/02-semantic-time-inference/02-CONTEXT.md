# Phase 2: Semantic-Time Inference - Context

**Gathered:** 2026-07-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Make `call_extern` (`python/triton/experimental/gluon/language/_semantic.py:250`) obtain the CUDA-inferred **element type and shape** from the Phase-1 inference hook at IR-build (semantic) time, build the `ttg.extern_call` result type from those (not from `first_input`), and leave **layout reconciliation** (`convert_layout`, native→user) to the already-working llir-stage C++ patch step (`tritonPatchExternCallResultTypes`, `python/src/clang_compiler.cc:1091`).

**In scope:** New inference-only C++ path (`inferReturnType` + `SuspendedCudaCompiler.infer()` binding); hook `infer_result()` consumption in `call_extern`; a `PlaceholderLayout` probe type in `tt_plugin.cu` enabling layout-independent dtype+shape deduction; per-result `(scalar, shape)` → `result_layouts` mapping; result-count mismatch guard; regression preservation of the 4 existing tests.

**Out of scope (this phase):**
- New shape/dtype-changing E2E test → Phase 3 (TEST-01)
- Full Python `TensorParameter`→`DistributedLinearLayout` round-trip (collapsed away — only `scalar`+`shape` used at semantic time)
- Making `result_layout` optional/auto-derived (AUTO-01, deferred)
- Full `Fp64` support (FP64-01, deferred; f64 already guarded in Phase 1)
- Moving all layout reconciliation into the frontend / retiring the C++ patch step
- `assert_no_conv` semantics rework (behavior unchanged; C++ patch already raises when a layout conversion is needed under `assert_no_conv=True`)

**Critical constraint:** The 4 existing extern-call tests (elementwise_add, intra_warp_add_sibling, reduce, split_add tuple) MUST pass **unchanged** — for those cases inferred (scalar, shape) equals the current `first_input`-derived (scalar, shape), so the built op type is identical.

</domain>

<decisions>
## Implementation Decisions

### infer_result Contract & Data Flow (INFER-01, INFER-02, INFER-03)
- **D-01:** Add a **new inference-only C++ method** on `CUDACompiler` — e.g. `inferReturnType(requests) -> ReturnTypes` — that runs ONLY the type-inference phase (`EvaluateFunctionReturnType` / `TypeInspector`) on the already-parsed, suspended (parked-in-`HandleTranslationUnit`) compiler, WITHOUT emitting bitcode. Add a matching `SuspendedCudaCompiler.infer()` pybind11 binding. This mirrors the Phase-1 D-02 two-method design (infer at semantic time / `compile_bitcode` at llir stage) and keeps inference cleanly separated from codegen. `compile_bitcode` later reuses the same parked AST for codegen (single parse preserved — INFER-07).
- **D-02:** `infer_result(func, arg_params, use_fast_math)` builds a `CudaFuncRequest` (symbol=func, param_types=[TensorParameter/ScalarType…], use_fast_math) — the **same** representation `_pre_compile_extern_calls` / `extract_extern_call_specs` (`clang_compiler.cc:756`) use at the llir stage. One conceptual code path for arg→spec conversion.
- **D-03 (layering):** The arg-handle → `TensorParameter`/`CudaFuncRequest` conversion lives in the **hook** (NVIDIA backend, where CUDA imports are allowed), NOT in `_semantic.py`. `call_extern` stays backend-agnostic: it passes arg handles/types + the opaque hook object (from `codegen_fns["infer_extern_call_result"]`) and receives back `(scalar, shape)` per result. Inference runs **before** the op is built, so the op is built exactly once with the CUDA-true dtype+shape.

### PlaceholderLayout Probe — the core mechanism (user's design)
- **D-04:** dtype+shape are guaranteed ready at `call_extern` time (every arg is a materialized `ttgl.tensor` with concrete `element_ty`+`shape`), but **layout is NOT** — Gluon `AutoLayout` (`_layouts.py:32`) resolves reg/lane/warp bases only after the ttgir layout-inference pass, i.e. after semantic time (`to_linear_layout` returns `AutoLayout` unresolved, `_semantic.py:437-438`).
- **D-05:** Resolve this by adding an **empty `PlaceholderLayout` type** to the `tt_plugin.cu` type framework (alongside `Shape`, `TensorLayout`, `Layout`, `Tensor`), plus an **implicit conversion** `Tensor<T, Shape, PlaceholderLayout>` → `Tensor<T, Shape, ConcreteLayout>` for any concrete `Layout`.
- **D-06:** At semantic time, always probe with `Tensor<T, Shape, PlaceholderLayout>` args — this deduces the return **dtype + shape** (which are layout-independent in the CUDA templates) via overload resolution + template deduction. If multiple candidates disagree on dtype/shape, **raise a clear error**.
- **D-07:** Full template inference + instantiation (which needs concrete layout bases) is deferred to when operand layouts are truly resolved — the existing llir-stage patch step — ideally embedding into Gluon's natural type-inference flow.

### Native-Layout vs convert_layout Split (INFER-04)
- **D-08:** **Semantic time fixes dtype + shape only.** `call_extern` builds `ttgl.distributed_type(inferred_dtype, inferred_shape, user_result_layout)` — the op's declared layout is the user's requested `result_layout` from the start.
- **D-09:** **The existing C++ patch step keeps layout + convert.** `tritonPatchExternCallResultTypes` (`clang_compiler.cc:1091`) continues to run full inference on now-concrete layouts at the llir stage, patch the op's native LAYOUT, and insert `ConvertLayoutOp` (native→user) exactly as it does today. Minimal blast radius; reuses the proven layout/convert path.
- **D-10:** Because semantic time now sets dtype+shape correctly, the C++ shape/dtype **hard-errors** (`clang_compiler.cc:1235` dtype, `1242` shape) become **never-fire safety asserts** rather than user-facing failures. (Keep them as defensive checks; they should no longer trigger for valid calls.)

### Round-Trip (collapsed)
- **D-11:** Phase 2 does **NOT** need the `TensorParameter`→`DistributedLinearLayout` Python round-trip. Only the inferred `scalar` (→dtype via `Fp32`=`fp32`, `Fp16`=`fp16`, `Bf16`=`bf16`, `Int32`=`int32`, `Int64`=`int64`) and `shape` are consumed at semantic time. The reg/lane/warp bases in the inferred `TensorParameter` are ignored at semantic time; the C++ patch step builds the native encoding via `buildEncodedType`/`LinearEncodingAttr` at the llir stage.

### Multi-Return & Regression (INFER-01/02, TEST-02 setup)
- **D-12:** `infer_result` returns a **list of `(scalar, shape)` per result**, ordered to match the user's `result_layouts` list (for `std::tuple<Tensor,…>`, this is the tuple element order that `EvaluateFunctionReturnType` already yields). `call_extern` zips: `result_types[i] = distributed_type(scalar_i→dtype, shape_i, result_layouts[i])`.
- **D-13:** If inferred result count != `len(result_layouts)`, **raise a clear error at semantic time** (mirrors the C++ result-count check at `clang_compiler.cc:1218`).
- **D-14:** Regression preserved by construction: same-shape/same-dtype functions yield inferred==declared → identical op type, zero behavior change; `reduce` (Shape<32,32>→Shape<32>) now infers `[32]` automatically and still matches the test's hand-supplied `[32]` `result_layout`, so it passes unchanged AND now works without hand-matching. **No test edits, no temporary dev-time guard.**

### the agent's Discretion
- Final name of the new C++ inference method and the `SuspendedCudaCompiler` binding (`inferReturnType` / `infer()` are working names).
- Exact `PlaceholderLayout` spelling and where the implicit conversion is declared in `tt_plugin.cu` (member conversion operator vs converting constructor) — provided it makes `Tensor<T,Shape,Placeholder>` deduce dtype+shape against the existing device functions.
- Precise error-message wording for the "candidates disagree on dtype/shape" and "result count mismatch" cases (intent locked; phrasing flexible).
- Exact mechanism for threading arg handles/types from `call_extern` into the hook (opaque object protocol), provided `_semantic.py` imports no NVIDIA backend code.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Reference implementation (full inference pipeline PoC)
- `/home/cicuvc/cs/project/nks/lab/cu_compiler_v2.cpp` — standalone PoC with `TypeBuilder`, `TypeInspector`, `FunctionResolver`, `CUDACompiler::EvaluateFunctionReturnType()`. Source of the inference-only path (D-01) and the placeholder/deduction pattern (D-04–D-07).
- `/home/cicuvc/cs/project/nks/lab/cu_compiler_v2.h` — headers/struct definitions for the above.

### The seam (frontend ↔ backend) and consumption points
- `python/triton/experimental/gluon/language/_semantic.py` — `call_extern` (line 250-279): consumes `infer_result`, builds result types from inferred (scalar, shape) + user `result_layout` (D-08, D-11, D-12). `_compute_result_shape` (line 18-28), `to_linear_layout` (line 427-440, note `AutoLayout` passthrough — D-04).
- `python/triton/experimental/gluon/language/_core.py` — `gl.call` user API (line 774-811).
- `third_party/nvidia/backend/compiler.py` — `InferExternCallResult` hook (line 190-252, `infer_result` stub at 220 to implement — D-02/D-03), `get_codegen_implementation` (hook creation ~line 340), `_pre_compile_extern_calls` (line 615, resumes compiler / emits bitcode).
- `python/triton/experimental/gluon/_runtime.py` — `make_ir` (line 47-103): `.cu` pre-scan, `create_and_suspend` at semantic time.

### CUDA compiler / bindings (where the new inference-only path lands)
- `python/src/clang_compiler.cc` / `.h` — `CUDACompiler` coroutine, `EvaluateFunctionReturnType`, `TypeInspector`, `FunctionResolver`, `BuildTensor`, `extract_extern_call_specs` (line 756), `tritonPatchExternCallResultTypes` (line 1091; dtype/shape hard-errors at 1235/1242, count check 1218, convert insert 1274 — D-09/D-10).
- `python/src/llvm.cc` — `ScalarType` enum (line 946), `TensorParameter` (line 955), `CudaFuncRequest` (line 1005), `SuspendedCudaCompiler` class (line 1029: `parse`/`compile_bitcode`; add `infer()` — D-01), `CudaFuncResult` (line 1011, carries `return_types`).

### CUDA device library (PlaceholderLayout goes here)
- `python/test/gluon/tt_plugin.cu` — type framework: `Shape` (line 7), `TensorLayout`/`BasisGroup`/`Layout` (line 30-79), `Tensor<T,TShape,TLayout>` (line 81-84); device fns `elementwise_add` (89), `intra_warp_add_sibling` (100), `add_bias` (118, uses `Sliced<0>`), `reduce` (127, Shape<32,32>→Shape<32>). Add `PlaceholderLayout` + implicit conversion here (D-05).

### Project docs
- `.planning/PROJECT.md` — locked decisions (result_layout required, infer-at-semantic-time, codegen_fns seam; CONCERNS.md partly outdated).
- `.planning/REQUIREMENTS.md` — INFER-01..05 (Phase 2 scope), TEST-* (Phase 3).
- `.planning/phases/01-seam-cleanup/01-CONTEXT.md` + `01-02-SUMMARY.md` — Phase-1 seam design (D-01..D-12 there), the `infer_result` stub, suspended-compiler + parse-counter machinery this phase builds on.
- `.planning/codebase/ARCHITECTURE.md` — "Return Type Inference from First Argument" anti-pattern (§Anti-Patterns) this phase removes; TensorParameter/clang AST bridge (§Key Abstractions).
- `.planning/codebase/CONCERNS.md` — coroutine/ABI fragility relevant to the suspend/resume path.
- `AGENTS.md` — build instructions, extern-call pipeline, return-type inference plan.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`InferExternCallResult` hook** (`compiler.py:190`) — already created, suspended, and keyed as `codegen_fns["infer_extern_call_result"]`; `infer_result()` is a ready stub to fill (currently raises `NotImplementedError`).
- **`SuspendedCudaCompiler`** (`llvm.cc:1029`) — parked-AST compiler with `parse()` + `compile_bitcode()`; add an `infer()` method alongside (D-01).
- **`CudaFuncResult.return_types`** (`llvm.cc:1011,1015`) — already carries inferred `TensorParameter`s out of `compile_bitcode`; the inference-only path returns the same.
- **`extract_extern_call_specs`** (`clang_compiler.cc:756`) — the canonical MLIR-operand → `TensorParameter`/`CudaFuncRequest` conversion to mirror for `infer_result`'s arg_params (D-02).
- **`tritonPatchExternCallResultTypes`** (`clang_compiler.cc:1091`) — the working layout-patch + `convert_layout` insertion; kept as-is for layout reconciliation (D-09).
- **`ScalarType`↔dtype-name mapping** — `Fp32`=`fp32`, `Fp16`=`fp16`, `Bf16`=`bf16`, `Int32`=`int32`, `Int64`=`int64` (D-11).

### Established Patterns
- Backend-agnostic frontend rule: `_semantic.py` must not import NVIDIA backend/CUDA code; all CUDA specifics arrive via the opaque `codegen_fns` hook (D-03).
- Two-method hook (Phase-1 D-02): infer at semantic time, `compile_bitcode` at llir stage — both backed by the same single parked parse (INFER-07 preserved).
- CUDA template return types are **layout-independent for dtype+shape** (see `reduce`/`add_bias` signatures) — the basis of the PlaceholderLayout probe (D-06).

### Integration Points
- `call_extern` (`_semantic.py:250`) — invoke hook, build result types from inferred (scalar, shape) (D-08, D-12).
- `InferExternCallResult.infer_result` (`compiler.py:220`) — fill the stub: build `CudaFuncRequest` from arg types, call `SuspendedCudaCompiler.infer()`, return per-result (scalar, shape) (D-02/D-03).
- `SuspendedCudaCompiler` (`llvm.cc:1029`) + `CUDACompiler` (`clang_compiler.cc`) — add inference-only `infer()`/`inferReturnType` (D-01).
- `tt_plugin.cu` type framework (lines 7-84) — add `PlaceholderLayout` + implicit conversion (D-05).

</code_context>

<specifics>
## Specific Ideas

- **User's PlaceholderLayout design (verbatim intent):** semantic layer / pre-ttgir cannot guarantee exact layouts, so (a) add an empty `PlaceholderLayout` in `tt_plugin.cu` with an implicit conversion from `Tensor<T,Shape,Placeholder>` to any concrete `Tensor<T,Shape,Layout>`; (b) at op-build time always probe with the Placeholder-constructed Tensor to fix dtype+shape first, erroring if multiple candidates give different dtype/shape; (c) run full template inference + instantiation only once operand layouts are fully determined — ideally embedded into Gluon's natural type-inference flow.
- **User's readiness clarification:** dtype/shape/layout are not all necessarily available before the ttgir conversion; dtype+shape are, but exact layout is not — hence the placeholder probe rather than requiring concrete arg layouts up front.
- Prefer failing loud (candidate disagreement, result-count mismatch) over silent fallback — continues the Phase-1 fail-loud stance.
- The C++ shape/dtype hard-errors become never-fire asserts — a signal that semantic-time inference, not the patch step, is now authoritative for dtype+shape.

</specifics>

<deferred>
## Deferred Ideas

- **AUTO-01** — Make `result_layout` optional / auto-derived from CUDA-inferred layout. Deferred (v2); user keeps `result_layout` explicit.
- **FP64-01** — Full `Fp64` support through the pipeline. Deferred; f64 already guarded (Phase 1).
- **Move all layout reconciliation to the frontend / retire the C++ patch step** — considered; rejected for Phase 2 in favor of the minimal split (semantic=dtype+shape, C++ patch=layout+convert). Could revisit if the two-convert-path boundary proves awkward.
- **assert_no_conv semantics rework** — skipped this discussion; current behavior (C++ patch raises when a layout conversion is required under `assert_no_conv=True`) is retained unchanged.
- **New shape/dtype-changing E2E test** — Phase 3 (TEST-01).

None of these belong in Phase 2.

</deferred>

---

*Phase: 2-semantic-time-inference*
*Context gathered: 2026-07-11*
