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

## Milestone: v1.1 — Shared Memory Interop

**Shipped:** 2026-07-23
**Phases:** 4 | **Plans:** 10 | **Tasks:** 29

### What Was Built
- C++ `SharedLinearLayout`/`SharedTensor<T,Shape,L>` device templates with OffsetBases/BlockBases NTTP carriers, mirroring the existing distributed `Layout`/`Tensor` pattern but with shared-memory (addrspace 3) addressing.
- Full clang AST round-trip: `TypeBuilder::BuildSharedTensor` (forward) → `TypeInspector::ParseSharedTensorType` (reverse), integrated into the v1.0 `FunctionResolver` for Sema template deduction with shared args.
- `ttg.extern_call` ODS relaxed to `AnyTypeOf<[TT_Tensor, TTG_MemDescType]>` — mixed tensor+memdesc operands with parse-verification lit tests; variant-based spec extraction (`std::variant<TensorSpecInput, SharedSpecInput>`) with `std::visit` JSON serialization.
- `gl.call()` frontend accepts `shared_memory_descriptor` arguments alongside tensors — `isinstance` relaxation, `PaddedSharedLayout` rejection guard, `memory_space` key in `arg_params`.
- Per-operand `ptr addrspace(3)` LLVM lowering — shared operands bypass alloca+store and use `getShmemAffineBase` GEP for subview offsets; distributed operands keep existing path in mixed arg lists.
- E2E verified: 6 new GPU tests (read_write, accumulate, 4× parametrized swizzle round-trip) + 6 existing extern-call regression + 6 lit + 1 pybind smoke = 19/19 pass, 0 fail.
- `gl.call()` scalar constexpr integer arg support (op attributes pipeline) and path portability fixes (LLVM_SYSPATH, CUDA_HOME env vars).

### What Worked
- **Phased architecture rollout:** Phase 4 (templates + AST round-trip) → Phase 5 (MLIR op + spec) → Phase 6 (wiring + lowering) → Phase 7 (E2E) — each phase built on a concrete foundation from the previous one, with its own verification gate.
- **std::variant over optional fields** (D-10): Clean type-level separation between tensor and shared spec inputs eliminated an entire class of null-field bugs.
- **Static test verification:** D-07 swizzle parity proven via 5 constexpr `static_assert` checks in a synthetic `.cu` — avoided the pre-existing coroutine segfault entirely.
- **TDD RED-GREEN cycles** in Phase 7 produced 4 atomic commits with clear test/implementation separation, and D-31 PTX landmine assertions guard against addrspace erasure in every shared-memory test.
- **Module-level attr for memory spaces:** `ttg.extern_call_arg_spaces` carried per-operand discriminators into the lowering pass — cleaner than inferring from types at the LLVM level.

### What Was Inefficient
- **LLVM dynamic-linking build issue** blocked GPU test execution through most of Phase 7 — self-compiled `libtriton.so` linked both static LLVM/Clang `.a` libraries AND `libLLVM.so.23.0git`, causing CLI option double-registration at `import triton` time. Investigation consumed significant time before path portability fixes resolved it.
- **Post-merge regressions** in Phase 6 (undefined `ttgl` NameError, clang reference-type assertion abort from `BuildSharedTensor`'s lvalue-ref wrapper) were caught late by the merge gate rather than pre-merge testing — could have been caught by running the test suite per-plan rather than per-phase.

### Patterns Established
- **NTTP carrier structs** (`OffsetBases`, `BlockBases`) for C++20 structural type requirements — reusable pattern for any non-type template parameter with rank-dependent dimensions.
- **`result_layout=[]`** for void-returning `gl.call()` device functions — empty list means zero result types, works naturally with existing result-type iteration.
- **`std::variant` + `std::visit`** for spec extraction with mixed operand types — type-safe alternative to tagged unions.
- **Frontend `isinstance` tuple guards** (`(ttgl.tensor, ttgl.shared_memory_descriptor)`) — explicit allow-list over deny-list for type safety.
- **PTX landmine assertions** (`assert 'ld.shared' in asm` / `assert 'st.shared' in asm`) as automated regression guard against lowering bugs that silently erase addrspace qualifiers.

### Key Lessons
1. **Build portability matters before verification:** The LLVM dynamic-linking issue could have been caught in Phase 4 if the test environment was exercised earlier. Pre-emptively fix these in the first phase of any new milestone.
2. **Static verification has its limits:** D-07 swizzle parity was proven statically, but the gap between static_assert and actual GPU execution was only closed in Phase 7 — 12 days later. Static checks are good for early confidence but don't replace E2E.
3. **Merge gates catch what per-plan testing misses:** Running the full test suite after each plan (not just each phase) would have caught the Phase 6 post-merge regressions earlier. Consider per-plan regression runs.
4. **The suspended-coroutine compiler pattern scaled cleanly** from v1.0 to v1.1 — adding `SharedTensorParameter` as a new variant case in `CudaFuncRequest::ParamTypes` was a one-line change at the dispatch site, proving the original architecture was well-factored.

### Cost Observations
- Model profile: adaptive; mode: yolo.
- 12 days, 85 commits, +15,501/-257 LOC across 102 files.
- Notable: Phase 7 verification alone consumed ~7 days once the LLVM build issue was encountered; the fix was ultimately simple (path portability) but diagnosis was expensive.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Key Change |
|-----------|--------|------------|
| v1.0 | 3 | Established gap-closure-after-verification loop and single-parse suspended-compiler seam |
| v1.1 | 4 | Phased architecture rollout (templates→MLIR→wiring→E2E); TDD RED-GREEN cycles; per-operand lowering via module attrs |

### Cumulative Quality

| Milestone | Tests | LOC Added |
|-----------|-------|-----------|
| v1.0 | 6 extern-call tests (all pass) | +5,760/-70 |
| v1.1 | 19 tests (12 GPU + 1 pybind + 6 lit, all pass) | +15,501/-257 |

### Top Lessons (Verified Across Milestones)

1. (v1.0) Suspend/resume beats double-parse when inference needs backend context.
2. (v1.1) Fix build portability early — infrastructure issues block everything downstream and can waste days of diagnosis.
3. (v1.1) The variant-based architecture (std::variant + std::visit) scaled cleanly; new argument types require minimal dispatch-site changes.
