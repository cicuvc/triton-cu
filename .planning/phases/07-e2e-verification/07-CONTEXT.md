# Phase 7: E2E Verification - Context

**Gathered:** 2026-07-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Full pipeline works end-to-end — shared memory read+write through `gl.call()` produces correct GPU results, swizzle layouts round-trip correctly, and all existing tests pass without regression. This is purely a verification/testing phase — all implementation (Phases 4-6) is complete.

**In scope:** Two new GPU tests for shared read+write (SHTEST-01); parametrized swizzle round-trip correctness test (SHTEST-02); full regression of 10 E2E tests + all Gluon lit tests (SHTEST-03); automated PTX grep for L-01 landmine (ld.shared/st.shared vs ld.generic); new CUDA device functions (`shared_accumulate`) in `tt_plugin.cu`.

**Explicitly NOT this phase:** New MLIR/pass changes (Phase 4-6 done); new frontend API (Phase 6 done); shared-memory return (SHRET-01, deferred); PaddedSharedLayout (deferred); auto-barriers (deferred). Covers requirements **SHTEST-01, SHTEST-02, SHTEST-03**.

</domain>

<decisions>
## Implementation Decisions

### Test function design (SHTEST-01)
- **D-24:** Two GPU tests for SHTEST-01:
  1. **Sequential read-write test**: kernel allocates shared memory → `gl.call("process_shared_2d")` (reads, multiplies by scale, writes back) → `gl.barrier()` → `shared_memory_descriptor.load()` with identity layout → store to output. Uses existing `process_shared_2d<T,TLayout>` (2D, `Shape<32,16>`, identity `SharedLinearLayout`, float32). Verifies write-back visibility to triton side.
  2. **Mixed-args test**: new CUDA function `shared_accumulate<T,N,TLayout>(SharedTensor&, const Tensor&)` — reads `shared[i]`, adds distributed tensor `val`, writes back. Kernel passes both shared descriptor AND a distributed tensor in a single `gl.call()`. Uses 1D shape (`Shape<256>`), identity `SharedLinearLayout`, float32. Verifies shared+distributed mixing at GPU runtime.
- **D-25:** `process_shared_2d` stays void-returning — no function signature changes. Verification of write-back is through `shared_memory_descriptor.load()` readback on the triton side, following `gl.barrier()`.
- **D-26:** New CUDA function `shared_accumulate` signature: `template<typename T, uint32_t N, typename TLayout> __device__ void shared_accumulate(SharedTensor<T, Shape<N>, TLayout>& shm, const Tensor<T, Shape<N>, TLayout>& val)` — iterates `[0..TLayout::REG_SIZE)`, `shm(i) += val.data[i]` (via `operator()` which requires the shared tensor shape to match the register count — identity layout ensures 1:1 mapping).

### Swizzle round-trip coverage (SHTEST-02)
- **D-27:** SHTEST-02 uses `@pytest.mark.parametrize` over 4 swizzle patterns covering isolated basis dimensions:
  1. **Identity** — `offset_bases=identity, block_bases=identity` (trivial, baseline)
  2. **Offset-only** — non-trivial offset basis (e.g., XOR bit-permutation), identity block basis
  3. **Block-only** — identity offset basis, non-trivial block basis
  4. **Full XOR** — non-trivial both offset and block bases (realistic swizzle)
  
  Test kernel: allocates shared memory with each swizzled `SharedLinearLayout` → `gl.call` a write-only function via `SharedTensor` at specific logical indices → `gl.barrier()` → `shared_memory_descriptor.load()` with identity layout → store to output.
- **D-28:** Verification: Python-side reference simulates swizzle logic (replicating `SharedLinearLayout::evaluate()`) to compute expected values — maps each logical (row, col) → swizzled byte offset → expected value. Pytest `torch.testing.assert_close` against kernel output.

### Reference computation strategy
- **D-29:** All 3 new tests follow the existing `test_extern_call.py` pattern: kernel stores results to output tensor via `gl.store()`, pytest computes expected values with `torch` on CPU side, `torch.testing.assert_close` compares. No in-kernel assertions, no special reference-passing — standard triton test pattern.

### Regression scope (SHTEST-03)
- **D-30:** Regression run:
  - **All 10 E2E tests**: 6 existing `test_extern_call.py` tests + 4 `test_shared_tensor.py` tests (Phase 4, pybind/clang round-trip) + new Phase 7 shared-memory tests
  - **All Gluon lit tests**: 5 original (`auto_encoding`, `infer_coalesced_encoding`, `inlining`, `invalid_auto_encoding`, `invalid_infer_coalesced_encoding`) + Phase 6 `extern-call-shared-args.mlir` lit test
  - Run as: `pytest python/test/gluon/test_extern_call.py python/test/gluon/test_shared_tensor.py -n 8` + `cd $BUILD_DIR && ninja triton-opt && lit -v test/Gluon/`

### L-01 landmine verification
- **D-31:** L-01 (Phase 6, AS3 pointer erasure): automated PTX grep in the shared-memory test kernel's compiled output. After each SHTEST-01/02 test run, access `compiled.asm["ptx"]` and assert `"ld.shared" in ptx or "st.shared" in ptx` for any shared-memory gl.call. Absence of `ld.generic` on shared addresses confirms D-17's direct AS3 pointer pass is effective. Recorded in CONTEXT.md as a persistent concern — any future change that routes shared pointers through stack slots could fire this.

### Locked upstream (carried forward — do not re-decide)
- All Phase 4 decisions: `SharedTensor` zero-length array storage (D-03), `operator()` returning `T&` (D-04), `SharedTensor&` mutable-reference convention (D-05), `SharedLinearLayout` distinct from distributed `Layout` (D-05)
- All Phase 5 decisions: `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>` (D-09), `std::variant<TensorSpecInput, SharedSpecInput>` (D-10)
- All Phase 6 decisions: `SharedTensorParameter` degenerate bases at semantic time (D-12), `LangAS::cuda_shared` addrspace qualifier (D-15), per-operand shared-vs-distributed lowering branch (D-16), direct AS3 ptr pass bypassing alloca+store+load (D-17), single-parse guard (compiler.py:683), `memory_space` dict key (D-14)
- Landmine L-01: AS3 pointer erasure through memory — D-17 mitigates but does not fix root cause (MemorySSA-class machinery needed)
- No auto-barriers — user places `gl.barrier()`; PaddedSharedLayout, dynamic shared, TMA interop all out of scope (REQUIREMENTS.md)

### the agent's Discretion
- Exact swizzle basis values for the 4 parametrized patterns in SHTEST-02 — researcher picks realistic non-trivial bases that exercise independent bit positions
- Whether `shared_accumulate` iterates 0..N (shape size) or 0..TLayout::REG_SIZE (register count) — identity layout makes them equal for the 1D test, but REG_SIZE is correct for future non-trivial layouts
- Whether to use `num_warps=1` (simpler) or `num_warps=2` (exercises multi-warp shared memory) for the new tests — researcher to decide based on what the lowering currently supports

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & milestone intent
- `.planning/REQUIREMENTS.md` § "Verification" — SHTEST-01, SHTEST-02, SHTEST-03 acceptance detail; out-of-scope table (shared return, PaddedSharedLayout, dynamic shared, TMA, auto-barriers)
- `.planning/ROADMAP.md` § "Phase 7" — the 4 success criteria (read+write GPU test, swizzle bit-identical, 6 E2E + 5 lit regression)
- `.planning/PROJECT.md` — locked decisions, L-01 landmine description, `ScalarType`↔dtype mapping, layering constraints, build/test commands

### Prior phase decisions
- `.planning/phases/06-cuda-wiring-llvm-lowering-frontend-api/06-CONTEXT.md` — D-12..D-23 (degenerate-basis inference, addrspace(3) lowering D-15/D-17, per-operand branch D-16, frontend validation D-18/D-19), L-01 landmine, D-22/D-23 verification approach
- `.planning/phases/05-mlir-op-relaxation-spec-extraction/05-CONTEXT.md` — D-09..D-11 (ODS relaxation, variant SpecInput, spec JSON schema with memory_space/offset_bases/block_bases/alignment)
- `.planning/phases/04-c-templates-clang-ast-foundation/04-CONTEXT.md` — D-01..D-08 (SharedLinearLayout/SharedTensor design, zero-length array, operator(), D-07 swizzle parity)

### Test infrastructure & device code
- `python/test/gluon/test_extern_call.py` — existing 6 E2E tests (elementwise_add, intra_warp_add_sibling, reduce, split_add_tuple, reduce_f16_f32, gl_call_no_inference_hook_raises); patterns to follow for new tests (torch.testing.assert_close, @pytest.mark.parametrize, kernel launch via [(1,)], torch.cuda.synchronize)
- `python/test/gluon/tt_plugin.cu` — existing device functions; `write_shared_1d` (line 216), `process_shared_2d` (line 220), `SharedTensor` template (line 173); new `shared_accumulate` lands here
- `python/test/gluon/test_shared_tensor.py` — Phase 4 shared tensor tests (pybind round-trip, 4 tests); part of regression surface
- `test/Gluon/` — 5 original Gluon lit tests + Phase 6 `extern-call-shared-args` lit test; all included in regression
- `.planning/codebase/TESTING.md` — test framework conventions, run commands, fixture patterns
- `AGENTS.md` — build/run/test commands, compiler pipeline overview

### GPU kernel API surface
- `python/triton/experimental/gluon/language/_core.py:185-514` — `shared_memory_descriptor_type` / `shared_memory_descriptor` API: `load(layout)`, `store(value)`, `slice/permute/reshape` subviews, barrier via `gl.barrier()`
- `python/triton/experimental/gluon/language/_layouts.py:174-213` — Python `SharedLinearLayout` / `SwizzledSharedLayout` / `NVMMASharedLayout` — the layout objects used to construct shared-memory descriptors in tests

### LLVM lowering & PTX (L-01 verification)
- `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` — shared lowering branch (D-16), direct AS3 ptr pass (D-17); compiled PTX accessible via `compiled.asm["ptx"]` in Python
- `third_party/nvidia/backend/compiler.py:709-872` — `_pre_compile_extern_calls()`, `compiled.asm` dict construction

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`process_shared_2d<T,TLayout>`** (`tt_plugin.cu:220`): existing read-modify-write device function — directly reusable for SHTEST-01 sequential test. Takes `SharedTensor<T,Shape<32,16>,TLayout>&` + `T scale` (non-tensor scalar). No distributed tensor arg — keeps it simple.
- **`write_shared_1d<T,N,TLayout>`** (`tt_plugin.cu:216`): single-element write pattern — shows the `shm(0) = val` syntax; `shared_accumulate` follows the same pattern extended to a loop with distributed tensor addition.
- **Existing E2E test pattern** (`test_extern_call.py`): `@pytest.mark.parametrize`, `torch.set_default_device('cuda')`, `kernel[(1,)]`, `torch.cuda.synchronize()`, `torch.testing.assert_close` — directly copied for new tests.
- **`@gluon.jit` kernel template**: `gl.constexpr` layout, `gl.arange`, `gl.load`, `gl.call`, `gl.store`, `gl.barrier()` — all used in existing tests; `shared_memory_descriptor` allocation via `gl.alloc_shared(dtype, shape, layout)`.
- **`compiled.asm["ptx"]`** — Phase 3's `test_reduce_f16_f32` already accesses `compiled.asm["ttgir"]` to assert IR structure; same pattern for PTX grep (D-31).

### Established Patterns
- **Test isolation**: each test function is self-contained — allocates inputs, launches kernel, synchronizes, asserts. No shared state between tests.
- **Single-CTA kernels**: all existing tests use `[(1,)]` grid launch (1 CTA). Shared memory tests should follow this — multi-CTA shared memory requires `block_bases` which SHTEST-02 tests separately.
- **Reference on CPU**: torch tensors on CUDA device, reference computed in torch (eagerly on GPU or CPU), `assert_close` compares. No custom reference kernels.
- **Error handling**: no in-kernel asserts — all verification is post-hoc in pytest.

### Integration Points
- `tt_plugin.cu` — new `shared_accumulate` function added adjacent to existing `write_shared_1d`/`process_shared_2d` (line ~225).
- `test_extern_call.py` — new test functions added after existing 6 tests; follow same `pytestmark` and import pattern.
- `gl.alloc_shared(dtype, shape, layout)` — the API for allocating shared memory in Gluon kernels (need to verify exact name — check `_core.py` for shared allocation entry point).
- `gl.barrier()` — explicit synchronization barrier; placed after `gl.call` that writes shared memory and before `load()` that reads it back.

</code_context>

<specifics>
## Specific Ideas

- The `shared_accumulate` function iterates `REG_SIZE` elements (register count of the distributed tensor), not the full shape of the shared tensor — identity layout makes these equal for the 1D test case, but REG_SIZE is the correct looping bound for general layouts.
- SHTEST-01 sequential test uses a scalar `scale` parameter on `process_shared_2d` (pass `1.0` or `2.0`), which goes through the existing scalar-to-distributed conversion in the glue layer — tests the full arg lowering path without adding a distributed tensor arg.
- SHTEST-02 swizzle bases should be chosen so that only 1 warp is needed (N_WARPS=1) — multi-warp introduces warp-level synchronization concerns that are out of scope. If `num_warps=1`, block_bases are always identity (single CTA), and the test focuses purely on offset swizzle correctness.
- PTX grep (D-31) should check both `ld.shared` AND `st.shared` — the shared_accumulate test does a read-modify-write, so both load and store to shared memory must use the correct addrspace instruction.
- L-01 is a persistent concern — even though Phase 6 avoids it through direct AS3 ptr pass, future refactors could introduce a stack slot. The automated PTX assertion catches regressions.

</specifics>

<deferred>
## Deferred Ideas

None from this discussion — all gray areas resolved within phase scope.

Prior deferrals remain (carried from REQUIREMENTS.md):
- **SHRET-01**: Returning `shared_memory_descriptor` from `gl.call()` — requires shared-memory liveness tracking; future milestone
- **AUTO-01**: Make `result_layout=` optional/auto-derived — deferred from v1.0
- **FP64-01**: Full Fp64 pipeline support — deferred from v1.0
- **PaddedSharedLayout**: padding doesn't map to shared linear layout — out of scope for v1.1
- **Dynamic `extern __shared__`**: variable-size allocation — out of scope (fixed-shape only)
- **Auto-barriers**: user must place `gl.barrier()` manually — documented scope boundary
- **AS3 pointer preservation across store/reload** (L-01 root cause fix): MemorySSA-class machinery — recorded as landmine only

</deferred>

---

*Phase: 7-E2E Verification*
*Context gathered: 2026-07-21*
