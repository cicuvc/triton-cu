# Phase 3: Verification - Context

**Gathered:** 2026-07-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove the `gl.call()` return-type inference feature end-to-end and guard against regressions. Add one new E2E test in `python/test/gluon/test_extern_call.py` that exercises an extern call whose return **shape AND dtype** differ from the first argument — supplying only the final `result_layout` (no hand-computed shape/dtype) — plus a new `.cu` device function to back it. Confirm the 4 existing extern-call tests still pass and that MLIR/LLVM module verification holds.

**In scope:**
- New CUDA device function in `tt_plugin.cu`: an `f16 → f32` reduction (`Tensor<half, Shape<32,32>> → Tensor<float, Shape<32>>`), accumulating in f32.
- New pytest test in `test_extern_call.py` for that function: only `result_layout` supplied, GPU numeric check vs a torch reference, PLUS an assertion that the compiled kernel's inferred result type has f32 element type + `[32]` shape.
- Regression confirmation: all 4 existing tests (elementwise_add, intra_warp_add_sibling, reduce, split_add tuple) pass unchanged.
- Explicit verification gate: run `make test-lit` to confirm the lit suite is unaffected (documented as a separate regression gate).

**Out of scope (this phase):**
- Any change to inference logic, `_semantic.py`, `clang_compiler.cc`, or the hook — this is a test-only + device-library-addition phase. (If a real inference bug surfaces, that is a fix, tracked separately.)
- Making `result_layout` optional/auto-derived (AUTO-01, deferred).
- Full `Fp64` support (FP64-01, deferred; f64 already guarded in Phase 1).
- Additional dtype/shape permutations beyond the one shape+dtype test (bf16, int32→int64 were considered and deferred — see Deferred Ideas).

**Critical constraint:** The new device function and test must NOT alter the behavior of the 4 existing tests. The existing `reduce` (fp32, shape-only) stays untouched; the new function is a separate symbol.

</domain>

<decisions>
## Implementation Decisions

### New Test's Change Dimension (TEST-01)
- **D-01:** The new E2E test exercises **shape AND dtype together** in a single test — the strongest single proof. `reduce` already covers shape-only in fp32; the **dtype-inference path (INFER-02) currently has ZERO E2E coverage** (all 4 existing tests are `T → T` same-dtype). One test that changes both closes the INFER-02 gap and re-confirms INFER-01 simultaneously.

### New CUDA Device Function Design (drives TEST-01)
- **D-02:** Add a new device function to `tt_plugin.cu`: an **`f16 → f32` reduction** — `Tensor<half, Shape<32,32>, TArg> → Tensor<float, Shape<32>, TRes>`. Structurally mirrors the existing `reduce` (`tt_plugin.cu:142-151`): reuse/adapt the `TArg` (Shape<32,32>) and `TRes` (Shape<32>) layout aliases; the only changes are input element type `T = half` and return element type `T = float`, with the accumulator in **f32** (`Result.data[0]` is float, `+= Vals.data[i]` promotes half→float).
- **D-03:** Torch reference: `x.to(torch.float32).sum(1)` where `x` is the f16 input `[32,32]`. Input tensor created as `torch.randn((32,32), dtype=torch.float16)`; output `torch.empty((32,), dtype=torch.float32)`.
- **D-04:** The kernel passes only `result_layout=gl.SliceLayout(1, layout)` (the [32] final layout, exactly as the existing `reduce_kernel` does) — NOT a hand-computed shape or dtype. The f32 element type + [32] shape must come entirely from CUDA-side inference. This is the literal TEST-01 requirement.

### TEST-03 Verification Method
- **D-05:** **Explicit verify + run lit.** Beyond the GPU numeric check, the test additionally **asserts the compiled kernel's inferred result type** — inspect the compiled artifact (e.g. `kernel.asm['ttgir']` / compiled metadata) to confirm the `ttg.extern_call` (or its result tensor type) carries **f32 element type and `[32]` shape**. This proves inference produced the correct type *before* lowering, not merely that the number came out right.
- **D-06:** `make test-lit` is run as a **separate, documented regression gate** to confirm the lit suite is unaffected. (No MLIR/dialect source changes in this phase, so no lit regressions are expected — the run is a guard, not a fix target.) The existing pipeline's internal `llvm.verify_module` (which aborts on malformed IR) remains the backstop that a successful compile implies verification passed.

### Precision / dtype Matrix
- **D-07:** Numerical tolerance for the f16→f32 reduction check: **`rtol=1e-2, atol=1e-2`** via `torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)`. Standard for f16-input reductions; accumulation in f32 on-device keeps error well within this band across the 32 additions.
- **D-08:** Only the single `f16→f32` transition is exercised this phase. bf16→f32 and int32→int64 were explicitly considered and deferred to avoid scope creep (see Deferred Ideas) — one shape+dtype test satisfies TEST-01/TEST-02/TEST-03.

### the agent's Discretion
- Exact name of the new device function (working name: `reduce_f16` / `reduce_fp16_fp32`) and the new kernel/test (working names: `reduce_f16_kernel`, `test_reduce_f16_f32`).
- Whether to reuse the existing `TArg`/`TRes` layout aliases directly or introduce parallel aliases for the half-typed function, provided the layout structure matches the existing `reduce`.
- Exact mechanism/attribute to inspect for the "assert inferred result type" check (`kernel.asm['ttgir']` text match vs a structured metadata field) — intent locked (assert f32 + [32] appear as the extern_call result type); the precise API to read it is flexible.
- Precise wording of any assertion messages.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The test file and device library (where all Phase 3 work lands)
- `python/test/gluon/test_extern_call.py` — the E2E test file. Existing tests: `test_elementwise_add` (line 62), `test_intra_warp_add_sibling` (72), `test_reduce_different_shape` (82, the shape-only fp32 reduce to mirror), `test_split_add_tuple` (92), `test_gl_call_no_inference_hook_raises` (104). Add the new f16→f32 test here.
- `python/test/gluon/tt_plugin.cu` — CUDA device library. Existing `reduce` at line 142-151, its layout aliases `TArg`/`TRes` at 138-139, `PlaceholderLayout` at 81-83, `Tensor` template at 85-99. Add the new f16→f32 reduction device function here.

### Feature pipeline the test proves (read to understand what "inferred type" means)
- `python/triton/experimental/gluon/language/_semantic.py` — `call_extern` (line 250+): builds result type from CUDA-inferred (scalar, shape) + user `result_layout`. This is what D-04's "only result_layout supplied" relies on.
- `third_party/nvidia/backend/compiler.py` — `InferExternCallResult` hook + `_pre_compile_extern_calls`; where `llvm.verify_module` runs in the llir stage (D-06 backstop).
- `python/src/clang_compiler.cc` — `EvaluateFunctionReturnType`/`TypeInspector` (inference), `tritonPatchExternCallResultTypes` (layout patch + convert_layout). The dtype/shape hard-errors are now never-fire asserts (Phase-2 D-10).

### Project docs
- `.planning/PROJECT.md` — locked decisions; core value (result types match CUDA-inferred dtype/shape/layout).
- `.planning/REQUIREMENTS.md` — **TEST-01, TEST-02, TEST-03** (this phase's scope), lines 33-35 + traceability 78-80.
- `.planning/ROADMAP.md` — Phase 3 goal + 3 success criteria (lines 72-82).
- `.planning/phases/02-semantic-time-inference/02-CONTEXT.md` — Phase-2 decisions (D-01..D-14): dtype+shape inferred at semantic time, layout via C++ patch; `reduce` now infers `[32]` automatically. Explains why a dtype-changing test is the remaining coverage gap.
- `.planning/phases/01-seam-cleanup/01-CONTEXT.md` — the hook + suspended-compiler seam the pipeline runs on.
- `.planning/codebase/TESTING.md` — pytest patterns, `torch.testing.assert_close`, `pytestmark = skipif(not is_cuda())`, `make test-lit` / `make test-gluon` run commands, f16 capability gating example.
- `AGENTS.md` — build (`bash build.sh` → copy `libtriton.so`), run with `PYTHONPATH`, E2E test command.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`reduce_kernel` + `test_reduce_different_shape`** (`test_extern_call.py:36-44, 82-88`) — the closest analog: a shape-changing reduce with `result_layout=gl.SliceLayout(1, layout)` and torch ref `x.sum(1)`. The new f16→f32 test is a near-copy that changes input dtype to f16, output dtype to f32, ref to `x.to(float32).sum(1)`, and adds a tolerance + inferred-type assertion.
- **`reduce` device fn + `TArg`/`TRes` aliases** (`tt_plugin.cu:138-151`) — directly adaptable: same Shape<32,32>→Shape<32> layout structure; only element types (`T=half` in, `T=float` out) and accumulator dtype change.
- **`PlaceholderLayout` + `Tensor` converting ctor** (`tt_plugin.cu:81-99`) — the semantic-time dtype+shape probe path; the new function is deduced through it exactly like `reduce`, so no new inference machinery is needed.

### Established Patterns
- Every test: `torch.set_default_device('cuda')`, build inputs with `torch.randn`, `kernel[(1,)](..., num_warps=1)`, `torch.cuda.synchronize()`, then `torch.testing.assert_close`. Module skip via `pytestmark = pytest.mark.skipif(not is_cuda())`.
- Device functions are templated `__device__` fns returning `Tensor<T, Shape<...>, Layout>`; return-type inference reads their declared return type.
- CUDA template return types are **layout-independent for dtype+shape** — the basis of the PlaceholderLayout probe; the f16→f32 return type is deducible the same way as `reduce`.

### Integration Points
- `tt_plugin.cu` (append new device fn near `reduce`, line ~151) — the new CUDA symbol.
- `test_extern_call.py` (append new `@gluon.jit` kernel + `test_*` fn) — the new E2E test.
- Inferred-type assertion reads the compiled kernel artifact (e.g. `kernel.asm['ttgir']`) — no source changes to the compiler, only inspection in the test.

</code_context>

<specifics>
## Specific Ideas

- **User's chosen test shape:** one test that changes **both** shape and dtype (f16 [32,32] → f32 [32] reduction) — deliberately the single strongest proof rather than splitting into dtype-only + shape+dtype tests.
- **Why f16→f32 specifically:** the dtype-inference path (INFER-02) has zero E2E coverage today; a float reduction accumulating in f32 is a natural, numerically well-behaved way to change element type while also changing shape, reusing the existing `reduce` structure.
- **Assert the inferred type, not just the number:** the test must confirm the compiled result type is f32 + [32] (proving inference), not only that the GPU output matches — a correct number alone could mask a mis-typed-but-coincidentally-right path.
- **Lit run is a guard, not a target:** no dialect/MLIR source changes here, so `make test-lit` is expected green; running it is defensive confirmation for TEST-03.

</specifics>

<deferred>
## Deferred Ideas

- **bf16→f32 reduction test** — considered as an alternative dtype transition; deferred. Would need cc>=80 gating (RTX 5090 satisfies it) but adds no new coverage beyond the f16→f32 case for this milestone.
- **int32→int64 reduction test** — considered (exact integer accumulation, no tolerance); deferred. Exercises the int dtype-mapping path but is beyond the single-test scope for TEST-01.
- **Two separate tests (dtype-only + shape+dtype)** — considered; deferred in favor of one combined shape+dtype test that satisfies TEST-01 with less surface area.
- **AUTO-01** — Make `result_layout` optional / auto-derived. Deferred (v2).
- **FP64-01** — Full `Fp64` support. Deferred; f64 already guarded (Phase 1).

</deferred>

---

*Phase: 3-verification*
*Context gathered: 2026-07-12*
