# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Return Type Inference

**Shipped:** 2026-07-12
**Phases:** 3 | **Plans:** 8 | **Tasks:** 17

### What Was Built
- Frontend↔backend inference seam: `InferExternCallResult` hook via CUDA backend `codegen_fns`, with a suspended `CUDACompiler` that parses each `.cu` once at semantic time and resumes at the `llir` stage (parse-counter delta assertion proves single-parse).
- Semantic-time return-type inference: `call_extern` builds the `ttg.extern_call` result type from CUDA-inferred dtype + shape + native layout, reconciling to the user's `result_layout` via `convert_layout`.
- `PlaceholderLayout` + `LookupFunctionWithPlaceholderFallback` + `ExplicitTemplateArgs` path so fixed-layout functions (e.g. `reduce`) also derive shape/dtype from CUDA inference.
- Bundled cleanup: removed dead code, added `f64`/`fp64` `NotImplementedError` guards at both layers; hook-absent now raises a clear `RuntimeError`.
- E2E proof: `test_reduce_f16_f32` (f16→f32 reduction, shape AND dtype transition) — GPU output matches `x.to(float32).sum(1)` within rtol/atol=1e-2.

### What Worked
- Backend `codegen_fns` hook mirrored the existing `convert_custom_types`/`min_dot_size` pattern — clean layering with no NVIDIA imports leaking into the frontend.
- Verifier-flagged behavior-unverified truths were confirmed directly on GPU (RTX 5090), closing the loop rather than leaving them as gaps.
- Gap-closure plans (02-04, 02-05) inserted post-verification kept the milestone honest instead of shipping SC1 partial.

### What Was Inefficient
- Phase 2 required post-verification gap closure (fixed-layout inference + hook-absent raise) that could have been anticipated in planning — the initial semantic-time plan under-covered fixed-layout functions.
- CONCERNS.md was partly outdated and had to be re-verified against code, costing early investigation time.

### Patterns Established
- Suspended-coroutine compiler with a parse-counter assertion as a durable single-parse guarantee.
- `PlaceholderLayout` as the mechanism for layout-independent template argument deduction.
- Defense-in-depth guards mirrored at both the frontend semantic layer and the CUDA backend layer.

### Key Lessons
1. When inference needs backend-only context (sm, resource_dir, LLVMContext), suspend/resume the compiler rather than parsing twice — measure the guarantee with a counter assertion.
2. Fixed-layout / concrete-template-arg cases are a distinct code path from placeholder-deduced ones; plan for both up front.
3. Verify "partly outdated" design docs against live code before trusting their gap analysis.

### Cost Observations
- Model profile: adaptive; mode: yolo.
- Notable: single-day milestone (2026-07-11 → 2026-07-12), 35 files, +5760/-70 LOC.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Key Change |
|-----------|--------|------------|
| v1.0 | 3 | Established gap-closure-after-verification loop and single-parse suspended-compiler seam |

### Cumulative Quality

| Milestone | Tests | Zero-Dep Additions |
|-----------|-------|--------------------|
| v1.0 | 6 extern-call tests (all pass) | — |

### Top Lessons (Verified Across Milestones)

1. (v1.0) Suspend/resume beats double-parse when inference needs backend context.
