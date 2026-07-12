# Phase 2: Semantic-Time Inference - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-11
**Phase:** 2-semantic-time-inference
**Areas discussed:** infer_result contract & data flow, Native-layout vs convert_layout split, PlaceholderLayout probe design, TensorParameter→DistributedLinearLayout round-trip, Multi-return & regression preservation

---

## infer_result contract & data flow

### How to obtain return TensorParameters at semantic time

| Option | Description | Selected |
|--------|-------------|----------|
| New inferReturnType C++ method | Inference-only method on CUDACompiler running EvaluateFunctionReturnType/TypeInspector on the parked AST, no codegen; + SuspendedCudaCompiler.infer() binding | ✓ |
| Reuse compile_bitcode early, cache bitcode | Call existing compile_bitcode at semantic time, cache bitcode, skip re-emit at llir | |

**User's choice:** New inferReturnType C++ method.
**Notes:** Cleanest separation; matches Phase-1 D-02 two-method design. compile_bitcode reuses the same parked AST for codegen, preserving single-parse (INFER-07).

### How call_extern produces arg_params

| Option | Description | Selected |
|--------|-------------|----------|
| Build CudaFuncRequest from arg tensor types | Reuse extract_extern_call_specs-style logic (symbol + param_types + use_fast_math), one code path | ✓ |
| Pass dtype/shape/layout, hook assembles | Frontend passes lighter args; hook builds TensorParameter/CudaFuncRequest internally | |

**User's choice:** Build CudaFuncRequest from arg tensor types.
**Notes:** Consistent with existing llir-stage spec extraction; one conceptual conversion path.

### Where arg→TensorParameter conversion happens (layering)

| Option | Description | Selected |
|--------|-------------|----------|
| Extract specs from the built MLIR op (C++) | Build op with placeholder types first, then walk operands to make requests, infer, rebuild | |
| Hook converts arg handles→spec, infer before op build | Hook (NVIDIA backend) converts handles→TensorParameter; call_extern stays backend-agnostic; infer before op build | ✓ |

**User's choice:** Option 2 (hook converts, infer before op build).
**Notes:** User asked whether all operand type/shape/layout are ready at op build. Investigation: dtype+shape always ready (materialized tensors); layout NOT always ready (AutoLayout resolves after ttgir layout-inference pass, to_linear_layout returns AutoLayout unresolved at _semantic.py:437-438). This drove the PlaceholderLayout design below.

---

## PlaceholderLayout probe design

**User's proposal (free-text):** Since the semantic layer / pre-ttgir cannot guarantee exact layouts: (a) add an empty PlaceholderLayout type to the tt_plugin.cu framework with an implicit conversion Tensor<T,Shape,Placeholder> → Tensor<T,Shape,Layout>; (b) at op build always probe with the Placeholder-constructed Tensor to fix dtype+shape first, erroring if multiple candidates disagree on dtype/shape; (c) run full template inference + instantiation only once operand layouts are fully determined, ideally embedded into Gluon's natural type-inference flow.

### How semantic time handles unresolved (AutoLayout) arg layouts

| Option | Description | Selected |
|--------|-------------|----------|
| Infer shape/dtype always; layout only if concrete | Use guaranteed-ready dtype+shape; defer layout to llir patch step when args are AutoLayout | ✓ (superseded by user's PlaceholderLayout design, same intent) |
| Require concrete arg layouts, else raise | Force user to resolve layouts before gl.call | |
| Assume concrete, no special handling | Least code, fragile | |

**User's choice:** PlaceholderLayout probe (own design) — realizes the "infer dtype/shape always, defer layout" intent via a CUDA-side placeholder type + implicit conversion.

---

## Native-layout vs convert_layout split

| Option | Description | Selected |
|--------|-------------|----------|
| Semantic fixes dtype+shape; C++ patch keeps layout+convert | Semantic builds op with inferred dtype+shape + user result_layout; existing tritonPatchExternCallResultTypes patches layout + inserts convert_layout at llir stage; C++ hard-errors become never-fire asserts | ✓ |
| Move all reconciliation to frontend, retire C++ patch | Do dtype+shape AND layout at semantic time; emit frontend convert_layout; retire C++ patch convert | |

**User's choice:** Semantic fixes dtype+shape; C++ patch keeps layout+convert.
**Notes:** Minimal blast radius; reuses the proven layout/convert path. Resolves the two-convert-path overlap.

---

## TensorParameter→DistributedLinearLayout round-trip

| Option | Description | Selected |
|--------|-------------|----------|
| Only scalar+shape used at semantic time; no Python round-trip | Consume only inferred scalar (→dtype) + shape; ignore bases; C++ patch builds native encoding at llir | ✓ |
| Build DistributedLinearLayout in Python too | Reverse-path build native layout in Python, emit frontend convert | |

**User's choice:** Only scalar+shape at semantic time; no Python layout round-trip.
**Notes:** Collapses the round-trip out of Phase 2 scope entirely.

---

## Multi-return & regression preservation

### Multi-return mapping

| Option | Description | Selected |
|--------|-------------|----------|
| Per-result (scalar,shape) list zipped with result_layouts | infer_result returns ordered (scalar,shape) list; zip with result_layouts; count-mismatch raises at semantic time | ✓ |
| Single-return only infers; tuples stay on old path | Smaller change, inconsistent single vs multi | |

**User's choice:** Per-result (scalar,shape) list zipped with result_layouts.

### Regression preservation

| Option | Description | Selected |
|--------|-------------|----------|
| Inferred==declared for existing cases → tests pass unchanged | Same-shape cases produce identical type; reduce infers [32] matching hand-supplied layout; no test edits | ✓ |
| Add temporary dev-time equality guard | Assert inferred==first_input during dev, then remove | |

**User's choice:** Inferred==declared → tests pass unchanged. No edits, no temporary guard.

---

## the agent's Discretion

- Final name of the new C++ inference method + SuspendedCudaCompiler binding (working names inferReturnType / infer()).
- PlaceholderLayout spelling and where/how the implicit conversion is declared in tt_plugin.cu.
- Error-message wording for candidate-disagreement and result-count-mismatch cases.
- Mechanism for threading arg handles/types into the hook (opaque object protocol), keeping _semantic.py backend-agnostic.

## Deferred Ideas

- AUTO-01 — optional/auto-derived result_layout (v2).
- FP64-01 — full Fp64 support (deferred; f64 guarded in Phase 1).
- Move all layout reconciliation to frontend / retire C++ patch step (rejected for Phase 2).
- assert_no_conv semantics rework (skipped; behavior retained).
- New shape/dtype-changing E2E test (Phase 3, TEST-01).
