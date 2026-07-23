# triton-cu: CUDA C++ Interop for gl.call()

## What This Is

triton-cu is a fork of [Triton](https://github.com/triton-lang/triton) that adds in-process CUDA C++ interop via `gl.call()` — Gluon kernels can call `__device__` template functions from `.cu` files, JIT-compiled through clang CodeGen and linked into the kernel at compile time.

- **v1.0** (shipped 2026-07-12): Return-type inference — `gl.call()` returns tensors with the CUDA-side-inferred element type, shape, and layout.
- **v1.1** (shipped 2026-07-23): Shared memory interop — Gluon `shared_memory_descriptor` buffers can be passed into device functions as a new `SharedTensor<dtype, shape, layout>&` parameter type with correct addrspace-3 lowering, read+write access, and full swizzle support.

## Current State

**Shipped:** v1.1 Shared Memory Interop — 4 phases, 10 plans, 29 tasks completed across 12 days. 19/19 tests pass (12 GPU E2E, 1 pybind smoke, 6 lit), zero regressions.

**Next milestone:** v1.2 or v2.0 — TBD. Candidates: shared memory return type (SHRET-01), auto-derived `result_layout` (AUTO-01), full Fp64 pipeline (FP64-01), or new feature surface.

## Core Value

`gl.call()` produces MLIR result types whose element type, shape, and layout match what the CUDA C++ `__device__` function actually returns — determined by clang overload resolution + template deduction + return-type inspection — so kernels compile with type-consistent downstream IR. Extended so Gluon `shared_memory_descriptor` buffers can be passed into device functions as `SharedTensor<dtype, shape, layout>&` with correct addrspace-3 lowering, validated in 12 GPU E2E tests.

## Requirements

### Validated

- ✓ In-process CUDA template instantiation for `gl.call()` extern calls — existing
- ✓ Coroutine-based CUDA compiler (`CUDACompiler`) with clang Sema/CodeGen — existing
- ✓ CUDA-side return-type inference plumbing: `TypeInspector`, `FunctionResolver`, `CUDACompiler::EvaluateFunctionReturnType()`, wired into `tritonCompileCuda` and returned via `compile_cuda_to_module` — existing
- ✓ `tritonPatchExternCallResultTypes()` rebuilds the `ttg.extern_call` op with the CUDA-inferred **layout** and inserts a `convert_layout` back to the user's declared layout — existing
- ✓ `std::tuple<Tensor,...>` multi-return support via `get_tuple_elem` extractors — existing
- ✓ `use_fast_math` per-function fast-math flag — existing
- ✓ E2E tests: elementwise add, intra-warp shuffle, reduce (shape change via manual `result_layout`), split_add tuple — existing (`test_extern_call.py`)
- ✓ Frontend↔backend inference seam: `InferExternCallResult` hook via `codegen_fns` + single-parse suspended `CUDACompiler` with parse-counter guard — v1.0 Phase 1
- ✓ Bundled bug fixes: dead code removed (`compiler.py:510-513`), `f64`/`fp64` raises `NotImplementedError` at both frontend and backend layers — v1.0 Phase 1
- ✓ **INFER-01/02**: CUDA-inferred return shape+dtype flows into `ttg.extern_call` result type — v1.0 Phase 2
- ✓ **INFER-03/04/05**: Inference at IR-build time, `result_layout=` remains final layout, fixed-layout resolve, hook-absent raise, bundled bugs — v1.0 Phase 2
- ✓ **TEST-01/02/03**: E2E reduce f16→f32, 6/6 tests pass, 5/5 lit pass — v1.0 Phase 3
- ✓ **SHTYPE-01/02**: `SharedLinearLayout` (OffsetBases/BlockBases NTTP carriers, `evaluate()`) and `SharedTensor<T,Shape,L>` (variadic `operator()` → `T&`) device templates compile as valid CUDA C++20 — v1.1 Phase 4
- ✓ **SHAST-01/02/03**: `SharedTensorParameter` structs + pybind11 binding, `TypeBuilder::BuildSharedTensor` forward, `TypeInspector::ParseSharedTensorType` reverse — full clang AST round-trip — v1.1 Phase 4
- ✓ **D-07 swizzle parity**: C++ `SharedLinearLayout::evaluate()` proven bit-identical to MLIR `LinearLayout` composition via 5 static_assert checks — v1.1 Phase 4
- ✓ **SHMLIR-01**: `ttg.extern_call` ODS relaxed to `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>` — mixed tensor+memdesc operands parse — v1.1 Phase 5
- ✓ **SHMLIR-02**: `extractExternCallSpecs()` uses `std::variant<TensorSpecInput, SharedSpecInput>` with `dyn_cast<MemDescType>` branch emitting shared-layout JSON — v1.1 Phase 5
- ✓ **SHAPI-01**: `gl.call()` accepts `shared_memory_descriptor` args alongside tensors — v1.1 Phase 6
- ✓ **SHWIRE-01**: Shared args wired through CUDA compilation — `infer_result()` degenerate `SharedTensorParameter`, `_pre_compile_extern_calls()` spec-consumption, `ttg.extern_call_arg_spaces` module attr, `BuildSharedTensor` `LangAS::cuda_shared` — v1.1 Phase 6
- ✓ **SHLOWER-01/02**: `ttg.extern_call` shared operands lower as `ptr addrspace(3)` (bypassing alloca+store) with subview offsets via `getShmemAffineBase` GEP — v1.1 Phase 6
- ✓ **SHTEST-01**: E2E GPU test — shared memory read+write through `gl.call()` with `gl.barrier()` synchronization — v1.1 Phase 7
- ✓ **SHTEST-02**: Swizzle-correctness test — 4 parametrized patterns, bit-for-bit round-trip via Python `evaluate_shared()` — v1.1 Phase 7
- ✓ **SHTEST-03**: All 6 existing tests + 6 lit tests pass unchanged — v1.1 Phase 7 (19/19 pass, 0 fail)

### Active

No active requirements. Fresh requirements will be defined during `/gsd-new-milestone`.

### Deferred / Future

- SHRET-01: Return a `shared_memory_descriptor` result from `gl.call()` (shared-memory return type)
- AUTO-01: Make `result_layout=` optional / auto-derived from the CUDA-inferred layout
- FP64-01: Full `Fp64` support through the entire pipeline
- Split the 1,396-line `clang_compiler.cc` (tech debt)
- PaddedSharedLayout shared encoding support

### Out of Scope

- Returning a `shared_memory_descriptor` from `gl.call()` (shared-memory result type) — deferred to SHRET-01
- Making `result_layout=` fully optional / auto-derived — deferred to AUTO-01
- Full `Fp64` support through the pipeline — separate effort
- Refactoring the coroutine/ABI machinery (x86-64-only `X64SysVABI`, stack-dangling lambda captures)
- Splitting the 1,396-line `clang_compiler.cc` — tech debt
- Parallel/multi-threaded CUDA compilation — scaling concern
- Fixing hardcoded LLVM/CUDA/clang-resource paths — build/toolchain concern
- `PaddedSharedLayout` shared encoding — doesn't map cleanly to shared linear layout
- Dynamic / `extern __shared__` variable-size shared allocation — requires CUDA dynamic-shared-memory ABI
- TMA / async-copy / mbarrier interop — separate feature surface
- Auto-inserted synchronization — user must place `gl.barrier()` explicitly

## Context

- **v1.1 shipped:** 4 phases (Phases 4-7), 10 plans, 29 tasks, 85 commits, +15,501/-257 lines across 102 files over 12 days (2026-07-12 → 2026-07-23).
- **Current test suite:** 19/19 pass (12 GPU E2E in `test_extern_call.py` + 1 pybind smoke + 6 lit), zero regressions from v1.0.
- **New in v1.1:** `SharedLinearLayout`/`SharedTensor<T,Shape,L>` C++ device templates, `SharedTensorParameter` struct + pybind binding, full clang AST round-trip (TypeBuilder ↔ TypeInspector), ODS-relaxed `ttg.extern_call` for mixed tensor+memdesc operands, variant-based spec extraction, frontend `shared_memory_descriptor` acceptance in `gl.call()`, per-operand `ptr addrspace(3)` LLVM lowering with subview-offset GEPs, `gl.call()` scalar constexpr arg support.
- **Key source locations:** `python/test/gluon/tt_plugin.cu` (device templates), `python/src/clang_compiler.cc`/`.h` (clang infrastructure), `python/src/llvm.cc` (pybind bindings), `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` (LLVM lowering), `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` (ODS), `third_party/nvidia/backend/compiler.py` (CUDA backend), `python/triton/experimental/gluon/language/_core.py`/`_semantic.py` (frontend API).
- **Build:** Never `pip install -e .`. Use self-compiled LLVM via `-DLLVM_SYSPATH=...`, clang as compiler, `bash build.sh`. `clang_compiler.cc` compiled `-fno-rtti`. Run with `PYTHONPATH` set to local tree.

## Constraints

- **Build**: Never `pip install -e .` (overwrites venv triton). Use self-compiled LLVM via `-DLLVM_SYSPATH=...`, clang as compiler, `bash build.sh`. `clang_compiler.cc` compiled `-fno-rtti`.
- **Tech stack**: C++ (clang/LLVM APIs, MLIR, Triton/TritonGPU dialects), Python (Gluon frontend, NVIDIA backend), CUDA C++ device templates.
- **Layering**: The Gluon semantic layer is backend-agnostic; CUDA-specific inference must reach it via the backend `codegen_fns` hook, not by importing NVIDIA backend code into the frontend.
- **Correctness**: Downstream MLIR must stay type-consistent; verify with `llvm.verify_module` and lit/pytest. No regressions in the 6 existing extern-call tests.
- **Performance**: Inference triggers a clang parse; must not double-compile. Reuse/cache the parsed `.cu` between semantic-time inference and llir-stage bitcode compilation.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep `result_layout` required (not auto-derived) | User wants explicit control of the final layout; smaller blast radius than making it optional | ✓ Good — shipped v1.0; auto-derive deferred to AUTO-01 |
| Infer shape/dtype/layout at semantic (IR-build) time | Only way to keep the op result type and all downstream consumers type-consistent | ✓ Good — validated E2E in v1.0 (`test_reduce_f16_f32`) |
| Reach CUDA inference from Gluon frontend via backend `codegen_fns` hook | Preserves frontend/backend layering; mirrors existing `convert_custom_types`/`min_dot_size` pattern | ✓ Good — seam built in Phase 1 |
| SharedTensor is argument-only for v1.1 | Returning shared memory is a larger scope; passing shared buffers into device fns covers the primary use case | ✓ Good — shipped v1.1; return deferred to SHRET-01 |
| New C++ SharedLinearLayout distinct from distributed Layout | Shared memory addressing (offset/block bases + swizzle) differs from distributed reg/lane/warp bases | ✓ Good — shipped Phase 4; D-07 swizzle parity proven |
| OffsetBases/BlockBases use RANK+N_BASES NTTP carrier structs | C++20 NTTP requires structural types with fixed-size arrays; matches existing BasisGroup pattern | ✓ Good — Phase 4 |
| Swizzle parity verified via static_assert in synthetic .cu | `parse()` success proves constexpr checks; avoids pre-existing coroutine crash | ✓ Good — 5 checks pass |
| `std::variant<TensorSpecInput, SharedSpecInput>` instead of optional fields | Cleaner type-level separation per D-10 | ✓ Good — Phase 5 |
| Per-operand memory-space discrimination in lowering | `getArgMemorySpaces` helper reads module attr; shared operands bypass alloca path | ✓ Good — Phase 6 |
| `result_layout=[]` for void-returning `gl.call()` | Empty list means zero result types; avoids special-casing void | ✓ Good — Phase 7 |
| `gl.call()` scalar constexpr integer args via op attributes | Avoids LLVM type inference ambiguity; separate pipeline from tensor/shared args | ✓ Good — Phase 7 |

---

*Last updated: 2026-07-23 after v1.1 Shared Memory Interop milestone — 4 phases shipped, 19/19 tests pass, shared memory interop complete.*
