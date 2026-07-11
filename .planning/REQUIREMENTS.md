# Requirements: triton-cu Return Type Inference for gl.call()

**Defined:** 2026-07-11
**Core Value:** `gl.call()` produces MLIR result types (element type, shape, layout) matching what the CUDA C++ `__device__` function actually returns, with type-consistent downstream IR.

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Inference Pipeline

- [ ] **INFER-01**: CUDA-inferred return **shape** flows into the `ttg.extern_call` result type. Functions returning a different shape than the first argument (e.g. `reduce`: `Shape<32,32>` → `Shape<32>`) compile without the user hand-computing the shape.
- [ ] **INFER-02**: CUDA-inferred return **dtype** flows into the result type. Functions changing element type (e.g. `f16`→`f32`) produce a result tensor with the correct element type.
- [ ] **INFER-03**: Inference runs at IR-build (semantic) time. The `ttg.extern_call` op result type and all downstream consumers (`gl.store`, arithmetic, etc.) stay type-consistent — no MLIR verification failures after lowering.

### Layout Reconciliation

- [ ] **INFER-04**: `result_layout=` remains the requested **final** layout. When the CUDA-native layout differs, a `convert_layout` reconciles CUDA-native → user layout (reusing the existing patch/convert path where appropriate).
- [ ] **INFER-05**: `assert_no_conv=True` still raises when a layout conversion would be required.

### Frontend/Backend Seam

- [ ] **INFER-06**: The Gluon semantic layer reaches CUDA-specific inference through the backend `codegen_fns` hook (mirroring `convert_custom_types`/`min_dot_size`), without importing NVIDIA backend code into the frontend. Interpreter/non-CUDA backends degrade gracefully.
- [ ] **INFER-07**: No redundant clang parse — inference at semantic time and bitcode compilation at `llir` stage do not double-compile the same `.cu` (reuse/cache, or documented single-parse path).

### Bug Fixes (bundled)

- [ ] **BUG-01**: Remove dead unreachable code at `compiler.py:510-513`.
- [ ] **BUG-02**: Decide and implement handling for `f64`/`fp64` → `Fp32` silent coercion (`compiler.py:542`): either raise a clear error at the API boundary or document the coercion explicitly.

### Verification

- [ ] **TEST-01**: New E2E test in `test_extern_call.py` exercising an extern call whose return **shape and/or dtype** differs from the first argument, WITHOUT the user manually matching that shape/dtype (only the final `result_layout` supplied).
- [ ] **TEST-02**: All 4 existing extern-call tests (elementwise_add, intra_warp_add_sibling, reduce, split_add tuple) still pass.
- [ ] **TEST-03**: `llvm.verify_module` passes after extern linking for the new and existing cases; lit suite unaffected.

## v2 Requirements

Deferred to future work. Tracked but not in current roadmap.

### Auto Layout

- **AUTO-01**: Make `result_layout=` optional — auto-derive the final layout from the CUDA-inferred layout when the user supplies nothing.

### Full Precision

- **FP64-01**: Full `Fp64` support through the entire pipeline (`ScalarType::Fp64`, `getQualTypeFromScalarType`, lowering, `Ctx.DoubleTy`).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Making `result_layout` fully optional/auto-derived | User chose to keep it as explicit final-layout request; smaller blast radius |
| Full Fp64 pipeline support | Separate effort; only a guard/decision (BUG-02) is in scope now |
| Refactoring coroutine/ABI machinery (x86-64-only, dangling captures) | Fragile but not required for this feature |
| Splitting the 1,396-line `clang_compiler.cc` | Tech debt, not this milestone |
| Parallel/multi-threaded CUDA compilation | Scaling concern, deferred |
| Fixing hardcoded LLVM/CUDA/clang-resource paths | Build/toolchain concern, deferred |
| `EvaulateConstantTemplateNTTP` typo/UB hardening | Cosmetic + defensive; touch only if it blocks INFER work |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFER-06 | Phase 1 | Pending |
| INFER-07 | Phase 1 | Pending |
| BUG-01 | Phase 1 | Pending |
| BUG-02 | Phase 1 | Pending |
| INFER-01 | Phase 2 | Pending |
| INFER-02 | Phase 2 | Pending |
| INFER-03 | Phase 2 | Pending |
| INFER-04 | Phase 2 | Pending |
| INFER-05 | Phase 2 | Pending |
| TEST-01 | Phase 3 | Pending |
| TEST-02 | Phase 3 | Pending |
| TEST-03 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-11*
*Last updated: 2026-07-11 after initial definition*
