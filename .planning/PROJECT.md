# triton-cu: Return Type Inference for gl.call()

## What This Is

triton-cu is a fork of [Triton](https://github.com/triton-lang/triton) that adds in-process CUDA C++ interop via `gl.call()` — Gluon kernels can call `__device__` template functions from `.cu` files, JIT-compiled through clang CodeGen and linked into the kernel at compile time. This project completes the **return-type inference** feature so `gl.call()` returns tensors with the CUDA-side-inferred element type, shape, and layout instead of naively copying them from the first argument.

## Core Value

`gl.call()` produces MLIR result types whose element type, shape, and layout match what the CUDA C++ `__device__` function actually returns — determined by clang overload resolution + template deduction + return-type inspection — so kernels that change dtype/shape/layout (e.g. reductions, dtype casts) compile correctly with downstream IR that stays type-consistent.

## Requirements

### Validated

- ✓ In-process CUDA template instantiation for `gl.call()` extern calls — existing
- ✓ Coroutine-based CUDA compiler (`CUDACompiler`) with clang Sema/CodeGen — existing
- ✓ CUDA-side return-type inference plumbing: `TypeInspector`, `FunctionResolver`, `CUDACompiler::EvaluateFunctionReturnType()`, wired into `tritonCompileCuda` and returned via `compile_cuda_to_module` — existing
- ✓ `tritonPatchExternCallResultTypes()` rebuilds the `ttg.extern_call` op with the CUDA-inferred **layout** and inserts a `convert_layout` back to the user's declared layout — existing
- ✓ `std::tuple<Tensor,...>` multi-return support via `get_tuple_elem` extractors — existing
- ✓ `use_fast_math` per-function fast-math flag — existing
- ✓ E2E tests: elementwise add, intra-warp shuffle, reduce (shape change via manual `result_layout`), split_add tuple — existing (`test_extern_call.py`)
- ✓ Frontend↔backend inference seam: `InferExternCallResult` hook via `codegen_fns` + single-parse suspended `CUDACompiler` with parse-counter guard — Validated in Phase 1: Seam & Cleanup
- ✓ Bundled bug fixes: dead code removed (`compiler.py:510-513`), `f64`/`fp64` raises `NotImplementedError` at both frontend and backend layers (no silent Fp32 coercion) — Validated in Phase 1: Seam & Cleanup

### Active

- [ ] **INFER-01**: CUDA-inferred return **shape** flows into the `ttg.extern_call` result type (functions returning a different shape than the first arg compile without the user hand-computing shape)
- [ ] **INFER-02**: CUDA-inferred return **dtype** flows into the result type (functions changing element type compile correctly)
- [ ] **INFER-03**: Inference runs at IR-build (semantic) time so op result types and all downstream consumers stay type-consistent — no MLIR verification failures
- [ ] **INFER-04**: `result_layout=` remains as the requested **final** layout; a `convert_layout` reconciles the CUDA-native layout → user layout
- [ ] **INFER-05**: Bundled bug fixes: remove dead code (`compiler.py:510-513`), decide on `f64`→`fp32` silent coercion guard
- [ ] **INFER-06**: New E2E test exercising a shape-and-dtype-changing extern call, plus existing 4 tests still pass

### Out of Scope

- Making `result_layout=` fully optional / auto-derived — deferred; user chose to keep `result_layout` as an explicit final-layout request (decision 2026-07-11)
- Full `Fp64` support through the pipeline — separate effort; only a guard/decision is in scope here
- Refactoring the coroutine/ABI machinery (x86-64-only `X64SysVABI`, stack-dangling lambda captures) — fragile but out of scope
- Splitting the 1,396-line `clang_compiler.cc` — tech debt, not this milestone
- Parallel/multi-threaded CUDA compilation — scaling concern, out of scope
- Fixing hardcoded LLVM/CUDA/clang-resource paths — build/toolchain concern, out of scope

## Context

- **The gap (corrected from CONCERNS.md, which is partly outdated):** The C++ patch step (`tritonPatchExternCallResultTypes`, `clang_compiler.cc:950`) already rebuilds the op with the inferred *layout* and inserts a `convert_layout`. The real remaining gap is that it **hard-errors** on shape/dtype mismatch (`clang_compiler.cc:1094-1104`) instead of adopting the CUDA-inferred shape/dtype.
- **Why the patch step can't fix shape/dtype:** `convert_layout` cannot change shape or dtype — only layout. The op's *declared* result type (built in `_semantic.py:250-266` from `first_input.dtype` + `_compute_result_shape`) is what all downstream kernel ops (e.g. `gl.store`) are typed against. If CUDA's true shape/dtype differs, downstream IR becomes inconsistent and MLIR verification fails — hence the current defensive error.
- **Chosen approach (decision 2026-07-11):** *Infer at semantic time.* Run CUDA type inference during IR building (in `call_extern`) so the op result type uses the CUDA-true shape+dtype+layout from the start; `result_layout` stays as the requested final layout with a `convert_layout` reconciling CUDA-native → user layout. Larger change but produces consistent downstream IR.
- **Layering challenge:** Inference requires a full clang parse of the `.cu` (the expensive step), which today happens in the `llir` backend stage where `sm`, `resource_dir`, include paths, and the `LLVMContext` all live. The backend-agnostic Gluon semantic layer (`GluonSemantic`) does not have those. The clean seam is the backend `codegen_fns` hook (`get_codegen_implementation`, consumed via `self.builder.codegen_fns[...]`) — the same mechanism `convert_custom_types`/`min_dot_size` use — so the CUDA backend can inject a callable the semantic layer invokes.
- **Round-trip available:** `to_linear_layout` (`gluon_ir.cc:371`) → `layoutToGluon` (`gluon_ir.cc:191`) provides MLIR encoding ↔ Python `DistributedLinearLayout`. Inferred `TensorParameter` (reg/lane/warp bases + shape) can build a `DistributedLinearLayout` for the op's native result type.
- **dtype strings:** `ScalarType` ↔ triton dtype names: `Fp32`=`fp32`, `Fp16`=`fp16`, `Bf16`=`bf16`, `Int32`=`int32`, `Int64`=`int64`, `Fp64`=`fp64`.
- **Environment:** self-compiled LLVM at a hardcoded path; build via `bash build.sh` → `build/libtriton.so` → copy to `python/triton/_C/libtriton.so`; run with `PYTHONPATH` set to local tree. GPU present (RTX 5090). E2E: `pytest python/test/gluon/test_extern_call.py`.

## Constraints

- **Build**: Never `pip install -e .` (overwrites venv triton). Use self-compiled LLVM via `-DLLVM_SYSPATH=...`, clang as compiler, `bash build.sh`. `clang_compiler.cc` compiled `-fno-rtti`.
- **Tech stack**: C++ (clang/LLVM APIs, MLIR, Triton/TritonGPU dialects), Python (Gluon frontend, NVIDIA backend), CUDA C++ device templates.
- **Layering**: The Gluon semantic layer is backend-agnostic; CUDA-specific inference must reach it via the backend `codegen_fns` hook, not by importing NVIDIA backend code into the frontend.
- **Correctness**: Downstream MLIR must stay type-consistent; verify with `llvm.verify_module` and lit/pytest. No regressions in the 4 existing extern-call tests.
- **Performance**: Inference triggers a clang parse; must not double-compile. Reuse/cache the parsed `.cu` between the new semantic-time inference and the existing `llir`-stage bitcode compilation where practical.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep `result_layout` required (not auto-derived) | User wants explicit control of the final layout; smaller blast radius than making it optional | — Pending |
| Infer shape/dtype/layout at semantic (IR-build) time | Only way to keep the op result type and all downstream consumers type-consistent; patch-step `convert_layout` cannot change shape/dtype | — Pending |
| Reach CUDA inference from Gluon frontend via backend `codegen_fns` hook | Preserves frontend/backend layering; mirrors existing `convert_custom_types`/`min_dot_size` pattern | ✓ Good — seam built in Phase 1 |
| Treat CONCERNS.md as partly outdated | Verified in code that the patch step already handles layout + convert_layout; real gap is shape/dtype hard-error | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-11 after Phase 1 (Seam & Cleanup) completion*
