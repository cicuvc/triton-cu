# Milestones

## v1.0 Return Type Inference (Shipped: 2026-07-12)

**Phases completed:** 3 phases, 8 plans, 17 tasks

**Key accomplishments:**

- Removed dead unreachable code and added defense-in-depth f64/fp64 NotImplementedError guards at both Gluon semantic and CUDA backend layers
- Inference-hook object (InferExternCallResult) exposed via CUDA backend codegen_fns; suspended CUDACompiler parses .cu once at semantic time and resumes at llir stage; parse-counter delta assertion proves single-parse; all 4 existing tests pass.
- PlaceholderLayout in tt_plugin.cu + BuildTensor placeholder-mode branch + CUDACompiler::inferReturnTypes inference-only method
- Pybind11 `SuspendedCudaCompiler.infer()` binding + `InferExternCallResult.infer_result()` filled hook calling CUDACompiler::inferReturnTypes
- call_extern consumes CUDA-inferred dtype+shape via infer_hook.infer_result() with first_input fallback for fixed-layout functions
- Fixed-layout `reduce` (and any function with concrete TArg/TRes layout params) obtains return dtype+shape from CUDA inference via PlaceholderLayout + ExplicitTemplateArgs fallback, removing the try/except RuntimeError workaround
- `call_extern` now raises a clear `RuntimeError` when the `infer_extern_call_result` hook is absent (non-CUDA backend) instead of silently falling through to the `first_input`-based fallback, with an automated test verifying the raise.
- f16→f32 CUDA reduction E2E test proving gl.call() return-type inference handles simultaneous shape AND dtype transitions — only result_layout supplied, f32 element type + [32] shape inferred from CUDA.

---
