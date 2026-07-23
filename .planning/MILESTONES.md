# Milestones

## v1.1 Shared Memory Interop (Shipped: 2026-07-23)

**Phases completed:** 4 phases, 10 plans, 29 tasks

**Key accomplishments:**

- Device-side SharedLinearLayout/SharedTensor templates, clang AST bridge structs, and Python SharedTensorParameter binding — establishing the type system for shared-memory interop
- Complete forward (TypeBuilder) and reverse (TypeInspector) clang AST round-trip for SharedTensor types, and integrate SharedTensorParameter into the CUDACompiler variant-dispatch paths
- GPU-free pytest harness verifying SharedTensorParameter round-trip, function resolution, and D-07 swizzle parity via static_assert — all 4 tests pass without GPU
- Relaxed ttg.extern_call ODS from Variadic<TT_Tensor> to AnyTypeOf<[TT_Tensor, TTG_MemDescType]> with parse-verification lit tests (SHMLIR-01)
- Variant-based data model with dyn_cast branch + std::visit serialization — eliminates crash on MemDescType operands and emits shared-layout JSON for Phase 6 consumption
- Enable gl.call() to accept shared_memory_descriptor arguments alongside tensors with frontend isinstance relaxation, PaddedSharedLayout rejection guard, and memory_space-keyed arg_params signaling
- Wired SharedTensorParameter through infer_result degenerate-basis, _pre_compile_extern_calls spec-consumption loops, ttg.extern_call_arg_spaces module attribute, and BuildSharedTensor LangAS::cuda_shared addrspace qualifier
- Shared-memory `ttg.extern_call` operands lower to `!llvm.ptr<3>` via `getShmemAffineBase` bypassing alloca, while distributed operands keep the existing `alloca+store+ptr` path — verified by mixed-arg lit test
- Two new `__device__` template functions added to tt_plugin.cu: shared_accumulate (4-template-param mixed-args accumulator) and write_swizzled_2d (2D identity-value writer for swizzle round-trip), with TDD file-content verification harness.
- Three new @gluon.jit kernels and three pytest functions added to test_extern_call.py: sequential read-modify-write (process_shared_2d), mixed shared+distributed accumulation (shared_accumulate), and 4-pattern swizzle round-trip with Python-side reference evaluator — comprehensive Phase 4-6 shared memory interop test coverage.

---

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
