# Phase 7: E2E Verification - Research

**Researched:** 2026-07-21
**Domain:** GPU end-to-end verification of CUDA C++ shared memory interop via `gl.call()`
**Confidence:** HIGH

## Summary

Phase 7 is a pure verification/testing phase — all implementation work from Phases 4-6 is complete. The work consists of: (a) adding two new GPU E2E tests to `test_extern_call.py` for shared-memory read+write through `gl.call()`, (b) a parametrized swizzle round-trip correctness test, (c) adding one new CUDA device function (`shared_accumulate`) to `tt_plugin.cu`, and (d) confirming zero regressions across all 10 E2E tests + 5 Gluon lit tests + 1 Phase 6 lit test.

The existing test infrastructure in `test_extern_call.py` (167 lines, 6 tests) provides a proven pattern: `@gluon.jit` kernel → `torch` inputs → launch `kernel[(1,)]` → `torch.cuda.synchronize()` → `torch.testing.assert_close`. The CUDA device code in `tt_plugin.cu` (310 lines) contains `process_shared_2d` (reusable for SHTEST-01) and `SharedTensor`/`SharedLinearLayout` templates. The shared memory API is `ttgl.allocate_shared_memory(dtype, shape, layout)` → `shared_memory_descriptor` → `smem.load(layout)` / `smem.store(val)`.

**Primary recommendation:** Follow the existing `test_extern_call.py` pattern exactly. All 3 new tests use `@gluon.jit` kernels with `[(1,)]` grid launch, `num_warps=1`, `torch.testing.assert_close` against CPU-computed reference. No new infrastructure needed.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-24:** Two GPU tests for SHTEST-01: (1) sequential read-write using existing `process_shared_2d`, (2) mixed-args using new `shared_accumulate` with both shared AND distributed tensor args
- **D-25:** `process_shared_2d` stays void-returning — write-back verified via `shared_memory_descriptor.load()` after `gl.barrier()`
- **D-26:** New `shared_accumulate` signature takes `(SharedTensor<T,Shape<N>,TLayout>&, const Tensor<T,Shape<N>,TLayout>&)` — iterates, does `shm(i) += val.data[i]`
- **D-27:** SHTEST-02 uses `@pytest.mark.parametrize` over 4 swizzle patterns: identity, offset-only, block-only, full XOR
- **D-28:** Python-side reference simulates `SharedLinearLayout::evaluate()` for expected-value computation
- **D-29:** Standard `torch.testing.assert_close` pattern — no in-kernel assertions
- **D-30:** Regression run: all 10 E2E tests (`test_extern_call.py` 6 existing + 4 new Phase 7 = 10 total) + `test_shared_tensor.py` 4 tests + all Gluon lit tests
- **D-31:** L-01 landmine — automated PTX grep via `compiled.asm["ptx"]` must find `ld.shared`/`st.shared` for shared-memory `gl.call()`

### the agent's Discretion
- Exact swizzle basis values for the 4 parametrized patterns in SHTEST-02 (resolved below)
- `shared_accumulate` loop bound: uses `TLayout::REG_SIZE` from the distributed tensor's layout (not `N` from shape — correct for future non-trivial layouts) [RESOLVED: see §CUDA Code Patterns]
- `num_warps=1` for all new tests (simpler; multi-warp would introduce synchronization complications out of scope) [RESOLVED: use `num_warps=1`]

### Deferred Ideas (OUT OF SCOPE)
- SHRET-01: Returning `shared_memory_descriptor` from `gl.call()`
- AUTO-01: Make `result_layout=` optional/auto-derived
- FP64-01: Full Fp64 pipeline support
- PaddedSharedLayout
- Dynamic `extern __shared__`
- Auto-barriers
- AS3 pointer preservation across store/reload (L-01 root cause)

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Shared memory allocation | Gluon Python API | Triton GPU compiler | `ttgl.allocate_shared_memory()` creates `memdesc` in MLIR; lowering produces shared memory ops |
| CUDA device function execution | Clang CodeGen (in-process) | PTX backend | `gl.call()` JIT-compiles `.cu` → bitcode → PTX, linked into kernel at compile time |
| Shared memory read/write in C++ | Device (GPU addrspace 3) | Clang AST | `SharedTensor<T,Shape,SharedLinearLayout>` maps logical indices → byte offsets in shared memory |
| Data validation & assertions | CPU (pytest/torch) | — | `torch.testing.assert_close` compares GPU output against CPU-computed reference |
| Barrier synchronization | Gluon Python API → PTX | — | `gl.barrier()` → PTX `bar.sync` instruction |
| PTX instruction verification | CPU (pytest) | — | `compiled.asm["ptx"]` string inspection for `ld.shared`/`st.shared` presence |

## Standard Stack

### Core (no version changes — existing infrastructure reused)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.12.11 | Test scripting | Existing project Python |
| torch | 2.7.1+cu128 | GPU tensor ops, reference computation | Existing project dep |
| triton (fork) | 3.5.0 (local) | Gluon kernel JIT compilation | This repository |
| pytest | (bundled) | Test runner, parametrize | Existing test infra |
| CUDA | 12.8 / sm_120 | GPU target (RTX 5090) | Existing GPU |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `triton._C.libtriton.llvm` | (compiled) | SharedTensorParameter, ScalarType | Only in `test_shared_tensor.py` (Phase 4 tests) |
| `triton._internal_testing.is_cuda` | (project) | Skip marker for non-CUDA envs | `pytestmark` in test files |

**Installation:** No new packages required. All deps are project-internal.

## Package Legitimacy Audit

> **No external packages are installed in this phase.** The phase adds test functions to existing test files and one CUDA device function to `tt_plugin.cu`. All dependencies are project-internal (pytest, torch, triton fork). No audit required.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

## Architecture Patterns

### System Architecture Diagram (SHTEST-01 Sequential Test)

```
Python test (pytest)
  │
  ├─► torch tensors (x, scale, out) on GPU
  │
  ├─► @gluon.jit kernel[(1,)] (num_warps=1)
  │     │
  │     ├─► ttgl.allocate_shared_memory(fp32, [32,16], shared_layout)
  │     │     └─► shared_memory_descriptor (MLIR: memdesc<32x16xf32, shared_linear, shared_memory>)
  │     │
  │     ├─► gl.call("tt_plugin.cu", "process_shared_2d", shm, scale)
  │     │     │
  │     │     ├─► _pre_compile_extern_calls(): JSON spec → SharedTensorParameter → clang CodeGen
  │     │     │     └─► bitcode: void process_shared_2d(SharedTensor<float,Shape<32,16>,SL>&, float)
  │     │     │
  │     │     └─► ExternCallOpToLLVM.cpp: shared arg → ptr addrspace(3) direct
  │     │           scale arg → alloca+store+ptr addrspace(0)
  │     │           └─► LLVM call @process_shared_2d(ptr<3>, float*)
  │     │
  │     ├─► gl.barrier()
  │     │     └─► PTX: bar.sync 0
  │     │
  │     ├─► result = shm.load(identity_layout)
  │     │     └─► loads shared memory → distributed tensor
  │     │
  │     └─► gl.store(out_ptr, result)
  │
  ├─► torch.cuda.synchronize()
  │
  └─► torch.testing.assert_close(out, reference)
```

### SHTEST-01 Mixed-Args Test

```
@gluon.jit kernel[(1,)] (num_warps=1)
  │
  ├─► vals = gl.load(x_ptr + idx)  // distributed tensor (1D, 256 elements)
  ├─► shm = ttgl.allocate_shared_memory(fp32, [256], SharedLinearLayout(identity))
  ├─► gl.call("tt_plugin.cu", "shared_accumulate", shm, vals)
  │     │  // shared_accumulate(SharedTensor&, const Tensor&): shm(i) += vals.data[i]
  │     └─► mixed args in single call: shm (addrspace 3) + vals (addrspace 0)
  ├─► gl.barrier()
  ├─► result = shm.load(distributed_layout)
  └─► gl.store(out_ptr, result)
```

### SHTEST-02 Swizzle Round-Trip

```
Python-side reference:
  for each (row, col) in shape:
    flatIndex = row * cols + col
    logical = SharedLinearLayout.evaluate(flatIndex)  // swizzle bit permutation
    byteOffset = dot(logical, byteStrides)
    expected[col][row] = f(byteOffset)  // deterministic value from offset

@gluon.jit kernel[(1,)]:
  shm = allocate_shared(shape, shared_layout=swizzled_SLL)
  gl.call("write_swizzled_2d", shm)  // writes per-element values via SharedTensor
  gl.barrier()
  result = shm.load(identity_layout)  // read back via identity SharedLinearLayout
  gl.store(out_ptr, result)

Assert: torch.testing.assert_close(out, expected)
```

### Recommended Project Structure (No new files — only modifications)
```
python/test/gluon/
├── test_extern_call.py     # ADD: test_shared_read_write (SHTEST-01), test_shared_accumulate (SHTEST-01), test_swizzle_round_trip (SHTEST-02)
├── tt_plugin.cu             # ADD: shared_accumulate device function (~line 226, after process_shared_2d)
└── test_shared_tensor.py    # UNCHANGED (regression)
```

### Pattern 1: Existing E2E Test Pattern (from `test_extern_call.py`)
**What:** Self-contained pytest function with `@gluon.jit` kernel, `torch` inputs, CTA launch, synchronize, assert_close.
**When to use:** All 3 new tests.
**Example:**
```python
# Source: python/test/gluon/test_extern_call.py:73-81
@pytest.mark.parametrize("BLOCK", [512])
def test_elementwise_add(BLOCK):
    torch.set_default_device('cuda')
    x = torch.randn((BLOCK,), dtype=torch.float32)
    y = torch.randn((BLOCK,), dtype=torch.float32)
    out = torch.empty_like(x)
    elementwise_add_kernel[(1,)](x, y, out, num_warps=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y)
```

### Pattern 2: compiled.asm PTX Access (from `test_lowerings.py`)
**What:** Capture the return value from kernel launch to access `compiled.asm["ptx"]` for instruction-level assertions.
**When to use:** D-31 (L-01 landmine) PTX grep in SHTEST-01/02 tests.
**Example:**
```python
# Source: python/test/gluon/test_lowerings.py:197-201
compiled = kernel[(1,)](out, num_warps=4, num_ctas=2)
ptx = compiled.asm["ptx"]
assert "ld.shared" in ptx or "st.shared" in ptx
```

### Pattern 3: Compiled IR Access (from `test_extern_call.py`)
**What:** Access `compiled.asm["ttgir"]` (or `"ptx"`, `"llir"`) for IR-level assertions.
**When to use:** D-31 PTX grep.
**Example:**
```python
# Source: python/test/gluon/test_extern_call.py:122,131-132
compiled = reduce_f16_kernel[(1,)](x, out, num_warps=1)
ttgir = compiled.asm["ttgir"]
assert "f32" in ttgir
```

### Pattern 4: Shared Memory Allocation + load/store (from `test_lowerings.py`)
**What:** `ttgl.allocate_shared_memory(dtype, shape, shared_layout)` → `smem.store(x)` / `y = smem.load(layout)`.
**When to use:** All new tests that use shared memory.
**Example:**
```python
# Source: python/test/gluon/test_lowerings.py:576-588
@gluon.jit
def kernel(x_ptr, y_ptr, shape: ttgl.constexpr, layout: ttgl.constexpr, shared_layout: ttgl.constexpr):
    offs = ttgl.arange(0, shape[0], layout=layout)
    x = ttgl.load(x_ptr + offs)
    smem = ttgl.allocate_shared_memory(x.dtype, shape, shared_layout)
    smem.store(x)
    y = smem.load(layout)
    ttgl.store(y_ptr + offs, y)
```

### Anti-Patterns to Avoid
- **In-kernel assertions:** Don't attempt to assert values inside @gluon.jit kernels — all verification is post-hoc in pytest via `torch.testing.assert_close`. [ASSUMED]
- **Multi-CTA kernels:** Don't use `num_ctas > 1` for shared-memory `gl.call()` tests. Phase 6's block_bases are accepted as NTTPs but `SharedLinearLayout::evaluate()` ignores them (block contribution is always zero). Multi-CTA shared-memory with non-trivial block_bases is a future feature. [VERIFIED: codebase, tt_plugin.cu:160-166]
- **PaddedSharedLayout for gl.call() shared args:** Phase 6's `call_extern()` raises `NotImplementedError` for `PaddedSharedLayout` (D-19). Only `SharedLinearLayout`/`SwizzledSharedLayout`/`NVMMASharedLayout` pass validation. [CITED: 06-CONTEXT.md D-19]
- **Accessing `compiled.asm` before kernel execution:** The compiled object is the return value of `kernel[...](...)` — must capture it at launch time. [VERIFIED: codebase, test_extern_call.py:122]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Shared memory byte-offset computation | Custom Python swizzle simulator | Replicate `SharedLinearLayout::evaluate()` logic in Python (XOR addition of basis rows) | The C++ `evaluate()` is the ground truth; Python reference must match exactly |
| Swizzle expected values | Ad-hoc value mapping | Deterministic function of byte offset: `value[row][col] = byteOffset` (or `row*100+col` for identity) | Makes expected values computable from layout without lookup tables |

## Runtime State Inventory

**Phase 7 is a verification/testing phase — no rename, refactor, or migration.** This section is omitted as per the verification protocol (rename/refactor phases only).

## Common Pitfalls

### Pitfall 1: Void-returning gl.call() not supported
**What goes wrong:** All 6 existing `test_extern_call.py` tests call functions that return tensors. `process_shared_2d` and `shared_accumulate` are `__device__ void`. If the Gluon frontend rejects `gl.call()` without a `result_layout=`, the test won't compile.
**Why it happens:** `call_extern()` in `_semantic.py` may require result types even for void functions.
**How to avoid:** Before writing tests, verify void-returning `gl.call()` works:
```python
# Use a null-like result or test via a C++ wrapper that returns a dummy tensor
# Fallback: wrap shared_accumulate in a C++ function that returns a status value
```
**Warning signs:** `TypeError: gl.call() missing required argument: 'result_layout'` or MLIR verification failure about missing result types. If void-returning is not supported, wrap `shared_accumulate` in a C++ function returning a 1-element dummy tensor.

### Pitfall 2: Swizzle basis dimension mismatch
**What goes wrong:** The `SharedLinearLayout` basis rows have rank matching the tensor shape, but building the wrong number of basis rows (too few/many bits) causes incorrect byte offsets.
**Why it happens:** `evaluate()` maps flatIndex bits to per-dim coordinates via XOR of basis rows. The number of basis rows must equal `ceil(log2(shape[0] * shape[1]))` for a complete mapping.
**How to avoid:** Verify the total number of basis rows equals the bit-width of `shape[0] * shape[1]`. For Shape<32, 16> (512 elements = 9 bits), use exactly 9 offset basis rows.
**Warning signs:** Some logical indices map to the same byte offset (lossy mapping), or some byte offsets are unreachable.

### Pitfall 3: L-01 Landmine — AS3 pointer erasure
**What goes wrong:** Future refactors could introduce a stack slot between the shared-memory pointer and the callee, causing the pointer to lose its addrspace(3) qualifier and silently collapse to generic addressing.
**Why it happens:** LLVM's MemorySSA machinery can re-materialize pointers through `alloca`/`store`/`load` if the shared pointer is treated identically to distributed pointers.
**How to avoid:** D-31's automated PTX grep acts as a regression guard — after each SHTEST-01/02 test, assert `ld.shared` or `st.shared` exists in the PTX (indicating correct addrspace-3 instructions were emitted). This is the Phase 6 D-17 mitigation: direct AS3 ptr pass bypassing alloca+store+load.
**Warning signs:** PTX shows `ld.global`/`st.global` or `ld.generic` instead of `ld.shared`/`st.shared` for shared memory accesses.

### Pitfall 4: GPU memory race on shared memory
**What goes wrong:** Without explicit `gl.barrier()` between the `gl.call()` write and the `shm.load()` read, threads within the CTA may not see the written values.
**Why it happens:** `gl.call()` executes asynchronously within the thread block. Without a barrier, some threads may read before others have written.
**How to avoid:** Always place `gl.barrier()` after `gl.call()` that writes shared memory and before any subsequent `shm.load()` or `gl.store()` that reads it.
**Warning signs:** Non-deterministic test failures, values sometimes zero, passes on some runs but fails on others.

## Code Examples

### SHTEST-01 Sequential Read-Write Test Kernel
```python
# Source: pattern from python/test/gluon/test_extern_call.py + python/test/gluon/test_lowerings.py:576-588
@gluon.jit
def shared_read_write_kernel(out_ptr, SCALE: gl.constexpr):
    shared_layout: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[[1,0],[2,0],[4,0],[8,0],[0,1],[0,2],[0,4],[0,8],[0,16]],
        block_bases=[], alignment=16
    )
    dist_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 2], [1, 1], [1, 0])

    shm = gl.allocate_shared_memory(gl.float32, [32, 16], shared_layout)
    gl.call("python/test/gluon/tt_plugin.cu", "process_shared_2d", shm, SCALE)
    gl.barrier()
    result = shm.load(dist_layout)
    gl.store(out_ptr, result)
```

### SHTEST-02 Swizzle Round-Trip Kernel (parametrized)
```python
# Source: pattern from python/test/gluon/test_extern_call.py:73-81
@gluon.jit
def swizzle_kernel(out_ptr, SHARED_LAYOUT: gl.constexpr, DIST_LAYOUT: gl.constexpr):
    shm = gl.allocate_shared_memory(gl.float32, [32, 16], SHARED_LAYOUT)
    gl.call("python/test/gluon/tt_plugin.cu", "write_swizzled_2d", shm)
    gl.barrier()
    result = shm.load(DIST_LAYOUT)
    gl.store(out_ptr, result)
```

### Python SharedLinearLayout::evaluate() Reference Implementation
```python
# Replicates tt_plugin.cu:160-166 evaluate() logic
# Source: code-verified from tt_plugin.cu:45-49 BasisGroup::evaluate() pattern
def evaluate_shared(flat_index: int, offset_bases: List[List[int]]) -> List[int]:
    """Simulate SharedLinearLayout::evaluate() — XOR-add basis rows for set bits."""
    rank = len(offset_bases[0])
    result = [0] * rank
    for bit_pos, basis_row in enumerate(offset_bases):
        if (flat_index >> bit_pos) & 0x1:
            for d in range(rank):
                result[d] ^= basis_row[d]
    return result

def byte_offset(logical_coords: List[int], shape: List[int], elem_size: int) -> int:
    """Convert logical (row, col) to byte offset."""
    # Row-major: offset = sum(coord[d] * stride[d]) * elem_size
    offset = 0
    for d in range(len(shape)):
        stride = elem_size
        for k in range(d + 1, len(shape)):
            stride *= shape[k]
        offset += logical_coords[d] * stride
    return offset
```

### shared_accumulate CUDA Device Function
```cpp
// Source: D-26 + tt_plugin.cu:216-225 pattern
// Add after process_shared_2d at ~line 226 in tt_plugin.cu
template<typename T, uint32_t N, typename SharedTLayout, typename TLayout>
__device__ void shared_accumulate(
    SharedTensor<T, Shape<N>, SharedTLayout>& shm,
    const Tensor<T, Shape<N>, TLayout>& val)
{
    #pragma unroll
    for (uint32_t i = 0; i < TLayout::REG_SIZE; i++)
        shm(i) += val.data[i];
}
```

### PTX Grep Pattern (D-31)
```python
# Source: pattern from python/test/gluon/test_lowerings.py:197-201
def test_shared_read_write():
    # ... kernel setup ...
    compiled = shared_read_write_kernel[(1,)](out, SCALE, num_warps=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected)

    # D-31: L-01 landmine — verify shared memory instructions in PTX
    ptx = compiled.asm["ptx"]
    assert "ld.shared" in ptx or "st.shared" in ptx, (
        f"L-01 LANDMINE: Expected ld.shared or st.shared in PTX but found neither.\n"
        f"AS3 pointer may have been erased through memory. First 200 chars:\n{ptx[:200]}"
    )
```

## Swizzle Reference: 4 Parametrized Patterns (SHTEST-02)

All patterns use 2D `Shape<32, 16>`, `num_warps=1`, `alignment=16`. Total flat index bits: log2(512) = 9. Each pattern has exactly 9 offset basis rows.

### Pattern 1: Identity (Baseline)
Standard row-major mapping — no bit permutation.

```
offset_bases = [
    [1, 0], [2, 0], [4, 0], [8, 0],         # col bits 0-3  (dim0 = inner)
    [0, 1], [0, 2], [0, 4], [0, 8], [0, 16],  # row bits 4-8  (dim1 = outer)
]
block_bases = []
```
**Behavior:** `evaluate(n)` returns `(n % 16, n // 16)` for a row-major layout.

### Pattern 2: Offset-Only XOR Swizzle
Swap bit 0 and bit 1 on the column dimension — adjacent pairs within each row are swapped in swizzle space.

```
offset_bases = [
    [2, 0], [1, 0], [4, 0], [8, 0],          # col bits 1,0 (swapped), 2,3
    [0, 1], [0, 2], [0, 4], [0, 8], [0, 16],  # row bits unchanged
]
block_bases = []
```
**Behavior:** `evaluate(0)=[0,0], evaluate(1)=[2,0], evaluate(2)=[1,0], evaluate(3)=[3,0]` — columns 1 and 2 are swapped.

### Pattern 3: XOR Cross-Dimension Swizzle
Swap a col bit with a row bit — col LSB flips row LSB and vice versa.

```
offset_bases = [
    [0, 1], [2, 0], [4, 0], [8, 0],          # col bit 0 → row bit 0 (cross-dim)
    [1, 0], [0, 2], [0, 4], [0, 8], [0, 16],  # row bit 0 → col bit 0 (cross-dim)
]
block_bases = []
```
**Behavior:** `evaluate(0)=[0,0], evaluate(1)=[0,1], evaluate(16)=[1,0], evaluate(17)=[1,1]` — interleaves lowest bits across dimensions.

### Pattern 4: Full XOR (All Bits Permuted)
Apply bit-reversal permutation within the flat index space — exercises all 9 basis rows in a non-identity configuration.

```
offset_bases = [
    [0, 16], [2, 0], [0, 8], [8, 0],          # bits 0,1,2,3: mixed dims (reverse-ish)
    [0, 4], [4, 0], [0, 2], [1, 0], [0, 1],   # bits 4-8: mixed dims
]
block_bases = []
```
**Behavior:** Every bit position maps to a different dimensional contribution than identity — exercises the general XOR composition code path.

**Verification strategy for all patterns:** The Python reference implements `evaluate_shared(flatIndex, offset_bases)` exactly replicating the C++ XOR-addition logic. For each `(row, col)` in the 32×16 tensor, compute `flatIndex = row * 16 + col`, then `logical = evaluate_shared(flatIndex)`, then `byteOffset = computeByteOffset(logical)`, and `expected[row][col] = byteOffset`. The kernel writes values via `gl.call("write_swizzled_2d", shm)` where `shm(i,j) = i*16+j`, reads back via identity layout, and pytest compares.

## CUDA Code Patterns

### Exact function to add: `shared_accumulate`
**Location:** `python/test/gluon/tt_plugin.cu`, after line 225 (after `process_shared_2d`)
**Pattern:** Follow `write_shared_1d` (line 216) + `process_shared_2d` (line 220) for argument convention (SharedTensor& first, mutable reference).
**Note on template parameters:** D-26 specifies `shared_accumulate<T,N,TLayout>` but `SharedLinearLayout` does not have `REG_SIZE`. The solution uses TWO layout parameters: `SharedTLayout` (for SharedTensor, auto-deduced) and `TLayout` (for Tensor's distributed layout, has `REG_SIZE`). The `TLayout::REG_SIZE` gives the distributed tensor's register count; for identity layouts in the 1D test, `REG_SIZE == N`.

```cpp
// Add at ~line 226 in tt_plugin.cu (after process_shared_2d closing brace)
template<typename T, uint32_t N, typename SharedTLayout, typename TLayout>
__device__ void shared_accumulate(
    SharedTensor<T, Shape<N>, SharedTLayout>& shm,
    const Tensor<T, Shape<N>, TLayout>& val)
{
    #pragma unroll
    for (uint32_t i = 0; i < TLayout::REG_SIZE; i++)
        shm(i) += val.data[i];
}
```

### Existing `process_shared_2d` (reusable, NO changes)
```cpp
// Source: python/test/gluon/tt_plugin.cu:220-225
template<typename T, typename TLayout>
__device__ void process_shared_2d(SharedTensor<T, Shape<32, 16>, TLayout>& shm, T scale) {
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 16; j++)
            shm(i, j) = shm(i, j) * scale;
}
```

### Existing `write_shared_1d` (reference for operator() usage)
```cpp
// Source: python/test/gluon/tt_plugin.cu:215-218
template<typename T, uint32_t N, typename TLayout>
__device__ void write_shared_1d(SharedTensor<T, Shape<N>, TLayout>& shm, T val) {
    shm(0) = val;
}
```

### `SharedTensor::operator()` return convention
```cpp
// Source: python/test/gluon/tt_plugin.cu:173-211
// Returns T& (mutable reference → read+write), accepts logical indices
__device__ T& operator()(auto... indices) {
    // row-major flatten → SharedLinearLayout::evaluate() → byte offset → data[byteOffset]
}
```

## API Surface

### Shared Memory Allocation
```python
# Function: ttgl.allocate_shared_memory (exported as gl.allocate_shared_memory)
# Source: python/triton/experimental/gluon/language/_core.py:613-630
shm = gl.allocate_shared_memory(gl.float32, [32, 16], shared_layout)
# Returns: shared_memory_descriptor with .load(layout), .store(value), .dtype, .shape
```

### Barrier Synchronization
```python
# Function: gl.barrier()
# Source: python/triton/experimental/gluon/language/_core.py:703-714
gl.barrier()  # CTA-wide sync
gl.barrier(cluster=True)  # cluster-wide sync (not used in Phase 7)
```

### Shared Memory Load/Store
```python
# Source: python/triton/experimental/gluon/language/_core.py:334-355
result_tensor = shm.load(distributed_layout)  # shared → distributed tensor
shm.store(source_tensor)                       # distributed → shared memory
```

### PTX Access (L-01 landmine)
```python
# Source: pattern from test_lowerings.py:197-201 + test_extern_call.py:122,131
compiled = kernel[(1,)](...)
ptx = compiled.asm["ptx"]      # PTX assembly string
ttgir = compiled.asm["ttgir"]  # TritonGPU IR string
llir = compiled.asm["llir"]    # LLVM IR string
# Available .asm keys: ["ttir", "ttgir", "llir", "ptx", "cubin"] and more
# Source: python/triton/compiler/compiler.py:424-429
```

### gluon.jit Kernel Pattern
```python
@gluon.jit
def my_kernel(ptr1, ptr2, MY_CONST: gl.constexpr):
    # ... kernel body ...
    pass

# Launch: 1 CTA (single block)
my_kernel[(1,)](tensor1, tensor2, MY_CONST_VALUE, num_warps=1)
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing, no config file changes) |
| Config file | No separate config — `pytestmark` in `test_extern_call.py` + `__init__.py` |
| Quick run command | `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python -m pytest python/test/gluon/test_extern_call.py -x -s --tb=short` |
| Full suite command | `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python -m pytest python/test/gluon/test_extern_call.py python/test/gluon/test_shared_tensor.py -x -n 8` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SHTEST-01 (sequential) | `process_shared_2d` read-modify-write via gl.call(), barrier, load back | GPU E2E | `pytest test_extern_call.py::test_shared_read_write -x` | ❌ Wave 0 |
| SHTEST-01 (mixed) | `shared_accumulate` with shared+distributed args, barrier, load back | GPU E2E | `pytest test_extern_call.py::test_shared_accumulate -x` | ❌ Wave 0 |
| SHTEST-02 | 4 parametrized swizzle patterns round-trip correctly | GPU E2E | `pytest test_extern_call.py::test_swizzle_round_trip -x` | ❌ Wave 0 |
| SHTEST-03 (regression) | All 6 existing tests pass unchanged | GPU E2E regression | `pytest test_extern_call.py -k "not test_shared_read_write and not test_shared_accumulate and not test_swizzle_round_trip" -x` | ✅ (6 existing) |
| SHTEST-03 (lit) | All Gluon lit tests pass | Lit (no GPU) | `cd $BUILD_DIR && ninja triton-opt && lit -v test/Gluon/` | ✅ (5 existing) |
| SHTEST-03 (lit Phase 6) | Phase 6 extern-call-shared-args lit test passes | Lit (no GPU) | `cd $BUILD_DIR && ninja triton-opt && lit -v test/TritonGPU/extern-call-shared-args.mlir` | ✅ |
| D-31 (L-01) | PTX grep for ld.shared/st.shared | PTX assertion | Built into SHTEST-01/02 test bodies | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest python/test/gluon/test_extern_call.py -x --tb=short -k "test_name"` (single test)
- **Per wave merge:** Full GPU test suite: `pytest python/test/gluon/test_extern_call.py python/test/gluon/test_shared_tensor.py -x -n 8`
- **Phase gate:** Full GPU test suite green + lit suite green

### Wave 0 Gaps
- [ ] Test functions to add in `test_extern_call.py`:
  - `test_shared_read_write` — SHTEST-01 sequential
  - `test_shared_accumulate` — SHTEST-01 mixed args
  - `test_swizzle_round_trip` — SHTEST-02 parametrized (4 patterns)
- [ ] CUDA device function `shared_accumulate` — SHTEST-01 mixed args (in `tt_plugin.cu`)
- [ ] `write_swizzled_2d` device function (or reuse existing pattern) — SHTEST-02 (in `tt_plugin.cu`)
- [ ] D-31 PTX grep assertions — embedded in SHTEST-01/02 test bodies
- [ ] No new test framework config needed (reuses existing infrastructure)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | — |
| V3 Session Management | No | — |
| V4 Access Control | No | — |
| V5 Input Validation | Yes (indirect) | Shared memory descriptor validation in `gl.call()` — `PaddedSharedLayout` rejection guard (D-19) |
| V6 Cryptography | No | — |

### Known Threat Patterns for CUDA/nvptx

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| AS3 pointer erasure (L-01) | Tampering | D-31 PTX grep — catch addrspace qualifier loss in future refactors |
| Shared memory buffer overrun | Tampering | `shared_accumulate` loop bound = `REG_SIZE` (not `N`), preventing out-of-bounds on non-identity layouts |
| Data race on shared memory | Information Disclosure / Tampering | Explicit `gl.barrier()` between write and read (user-placed) |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| RTX 5090 GPU | All GPU tests | ✓ | sm_120 | — |
| CUDA | GPU, clang compilation | ✓ | 12.8 | — |
| Python 3.12 | Test runner | ✓ | 3.12.11 | — |
| torch | GPU tensor ops | ✓ | 2.7.1+cu128 | — |
| triton (local fork) | Gluon JIT | ✓ | 3.5.0 (local) | — |
| libtriton.so (build) | Compiled extension | ✓ | build/libtriton.so (1.2GB) | Must copy to python/triton/_C/ for PYTHONPATH dev |
| clang | Build (not used at test time) | ✓ | 23.0.0git | — |
| LLVM resource dir | test_shared_tensor.py (Phase 4) | ✓ | /media/.../llvm-data/install | — |
| pytest | Test runner | ✓ | (bundled) | — |

**Missing dependencies with no fallback:** None — all dependencies are available.

## File Manifest

| File | Action | Lines/Changes | Purpose |
|------|--------|---------------|---------|
| `python/test/gluon/tt_plugin.cu` | ADD ~15 lines after line 225 | New `shared_accumulate` device function + `write_swizzled_2d` (if needed) | CUDA device function for SHTEST-01 mixed-args test + SHTEST-02 swizzle test |
| `python/test/gluon/test_extern_call.py` | ADD ~200 lines after line 167 | 3 new test functions + 3 kernel functions | SHTEST-01 (2 tests) + SHTEST-02 (1 parametrized test) + D-31 PTX assertions |
| `.planning/phases/07-e2e-verification/07-RESEARCH.md` | CREATE | This file | Research for planner |

**Files NOT touched:**
- `python/test/gluon/test_shared_tensor.py` — regression only (read)
- `python/triton/experimental/gluon/language/_core.py` — read-only (API reference)
- `python/triton/experimental/gluon/language/_layouts.py` — read-only (SharedLinearLayout API)
- `test/Gluon/*.mlir` — lit tests, regression only (read)
- `test/TritonGPU/extern-call-shared-args.mlir` — lit test, regression only (read)
- `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` — read-only (lowering reference)
- `third_party/nvidia/backend/compiler.py` — read-only (compilation pipeline reference)

## L-01 Landmine Verification Protocol

**Trigger:** Run after every shared-memory `gl.call()` test (SHTEST-01 both tests, SHTEST-02 all 4 patterns).

**Check:**
```python
ptx = compiled.asm["ptx"]
assert "ld.shared" in ptx or "st.shared" in ptx, (
    "L-01 LANDMINE TRIGGERED: No ld.shared or st.shared in PTX. "
    "AS3 pointer may have been erased through memory. "
    "Check Phase 6 D-17 — direct AS3 ptr pass must be preserved."
)
```

**Background:** Phase 6's D-17 ensures shared-memory pointers are passed directly as `ptr addrspace(3)` to callees (bypassing alloca+store+load). If a future refactor introduces a stack slot, the pointer collapses to generic addressing, producing `ld.generic` instead of `ld.shared`. This PTX assertion catches that regression at test time.

## Build & Run Commands

**Build (if libtriton.so needs update after tt_plugin.cu changes):**
The `tt_plugin.cu` is compiled at JIT time by clang CodeGen (not part of the build). Build is only needed if `python/src/clang_compiler.cc` or other C++ source changes. For Phase 7, no build is needed (libtriton.so unchanged).
```bash
# Only if C++ source changes (not expected for Phase 7):
bash build.sh && cp build/libtriton.so python/triton/_C/libtriton.so
```

**Run new tests:**
```bash
PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" \
    python -m pytest python/test/gluon/test_extern_call.py -x -s --tb=short \
    -k "test_shared_read_write or test_shared_accumulate or test_swizzle_round_trip"
```

**Run full regression:**
```bash
PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" \
    python -m pytest python/test/gluon/test_extern_call.py \
    python/test/gluon/test_shared_tensor.py -x -n 8
```

**Run lit tests:**
```bash
cd build && ninja triton-opt && lit -v test/Gluon/ test/TritonGPU/extern-call-shared-args.mlir
```

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `gl.call()` supports void-returning device functions without requiring `result_layout=` | Common Pitfalls #1, API Surface | High — if void return not supported, need C++ wrapper returning dummy tensor; adds ~10 lines to tt_plugin.cu and changes kernel code |
| A2 | `ttgl.allocate_shared_memory(gl.float32, [32,16], layout)` API name is `allocate_shared_memory` (not `alloc_shared`) | API Surface | Low — verified in test_lowerings.py:584 |
| A3 | `shared_accumulate` separate template params (`SharedTLayout`, `TLayout`) is cleaner than single `TLayout` (which can't have both `REG_SIZE` and `evaluate()`) | CUDA Code Patterns | Low — if wrong, template fails to compile with clear error about `REG_SIZE`; easy to fix by changing to 2-param approach |
| A4 | `write_swizzled_2d` CUDA device function pattern (for SHTEST-02) follows `process_shared_2d` but iterates all (i,j) writing `i*16+j` values | File Manifest | Low — simple function with known pattern; any compile error is trivial to fix |
| A5 | `num_warps=1` avoids multi-warp synchronization issues for shared memory tests | Common Pitfalls | Low — 1 warp is simplest; if it doesn't exercise enough code paths, can raise to 2 in subsequent iteration |

## Open Questions

1. **Does gl.call() support void-returning device functions?**
   - What we know: All 6 existing tests use tensor-returning functions. `process_shared_2d` and `shared_accumulate` are `__device__ void`. The Phase 6 frontend (SHAPI-01) relaxed `gl.call()` to accept shared args but may still require result types.
   - What's unclear: Whether `call_extern()` in `_semantic.py` will reject a call without `result_layout=` (or without return type inference finding types).
   - Recommendation: Test with a minimal kernel first: `gl.call("tt_plugin.cu", "write_shared_1d", shm, 1.0)` and see if it compiles. If it fails, wrap `shared_accumulate` in a C++ function returning a `Tensor<T, Shape<1>, DummyLayout>` dummy. [ASSUMED]

2. **Does `gl.call()` compile when `tt_plugin.cu` uses separate template params for SharedTLayout vs TLayout?**
   - What we know: The existing `process_shared_2d<T, TLayout>` compiles fine. Adding a second layout parameter for the distributed tensor is a C++20 template deduction question.
   - What's unclear: Whether clang's template argument deduction will correctly infer both layout types from the function arguments.
   - Recommendation: Use explicit template parameters in `shared_accumulate` if deduction fails. The `_pre_compile_extern_calls()` path handles explicit template parameters via `ExplicitTemplateArgs`.

## Sources

### Primary (HIGH confidence — codebase verification)
- `python/test/gluon/test_extern_call.py` (167 lines) — existing 6 E2E test patterns, imports, compiled.asm access [VERIFIED: codebase]
- `python/test/gluon/tt_plugin.cu` (310 lines) — SharedTensor template, process_shared_2d, write_shared_1d, SharedLinearLayout::evaluate() [VERIFIED: codebase]
- `python/triton/experimental/gluon/language/_core.py:613-630` — allocate_shared_memory API [VERIFIED: codebase]
- `python/triton/experimental/gluon/language/_layouts.py:631-673` — SharedLinearLayout constructor (offset_bases, block_bases, alignment) [VERIFIED: codebase]
- `python/test/gluon/test_lowerings.py:197-201,576-588` — compiled.asm["ptx"] access pattern, allocate_shared_memory + store/load pattern [VERIFIED: codebase]
- `python/triton/compiler/compiler.py:424-429` — compiled.asm dict construction from metadata_group suffixes [VERIFIED: codebase]
- `test/TritonGPU/extern-call-shared-args.mlir` — Phase 6 mixed-args lit test confirming mixed shared+distributed lowering [VERIFIED: codebase]

### Secondary (MEDIUM confidence — CONTEXT.md decisions)
- `.planning/phases/07-e2e-verification/07-CONTEXT.md` — D-24 through D-31 decisions [CITED: 07-CONTEXT.md]
- `.planning/phases/06-cuda-wiring-llvm-lowering-frontend-api/06-CONTEXT.md` — D-17 (direct AS3 ptr pass), D-16 (per-operand branch), L-01 landmine [CITED: 06-CONTEXT.md]

### Tertiary (LOW confidence — training knowledge)
- `gl.call()` void-return support — assumed based on template not having return-type restrictions, not verified against `_semantic.py:call_extern()` this session [ASSUMED]
- `write_swizzled_2d` device function pattern — assumed to follow `process_shared_2d` iteration pattern, not yet written [ASSUMED]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies are project-internal, verified via `--version` checks and import tests
- Architecture: HIGH — patterns verified by reading all 6 source files; architecture diagram traced from existing code
- Pitfalls: MEDIUM — void-returning gl.call() support and shared_accumulate template deduction are assumed; L-01 is well-documented from Phase 6
- Swizzle basis values: HIGH — computed by tracing SharedLinearLayout::evaluate() XOR logic, verified against existing ParityLayout static_asserts in tt_plugin.cu:270-298

**Research date:** 2026-07-21
**Valid until:** 2026-08-21 (30 days — stable test infrastructure)
