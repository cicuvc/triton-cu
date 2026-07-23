# Phase 07: E2E Verification - Pattern Map

**Mapped:** 2026-07-21
**Files analyzed:** 2 (both modifications to existing files)
**Analogs found:** 2 / 2

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `python/test/gluon/tt_plugin.cu` (+15 lines) | CUDA device function | in-place-mutation (shared memory read-modify-write) | `write_shared_1d` (line 215) / `process_shared_2d` (line 220) | exact |
| `python/test/gluon/test_extern_call.py` (+200 lines) | pytest test | request-response (kernel launch → assert_close) | `test_elementwise_add` (line 73) / `test_reduce_f16_f32` (line 116) | exact |

## Pattern Assignments

### `python/test/gluon/tt_plugin.cu` — ADD `shared_accumulate` (in-place-mutation)

**Analog:** `write_shared_1d` (lines 215-218) and `process_shared_2d` (lines 220-225)

**Core pattern** (lines 215-225):
```cpp
// T, N, TLayout template convention + SharedTensor<T>& mutable-reference (D-05)
template<typename T, uint32_t N, typename TLayout>
__device__ void write_shared_1d(SharedTensor<T, Shape<N>, TLayout>& shm, T val) {
    shm(0) = val;
}

template<typename T, typename TLayout>
__device__ void process_shared_2d(SharedTensor<T, Shape<32, 16>, TLayout>& shm, T scale) {
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 16; j++)
            shm(i, j) = shm(i, j) * scale;
}
```

**New function location:** After line 225 (after `process_shared_2d` closing brace), before line 227 (`// =========================` comment).

**New function code** (follows RESEARCH.md §CUDA Code Patterns):
```cpp
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

**Key conventions matched:**
- `__device__ void` return (non-tensor-returning, D-25/D-26)
- `SharedTensor<T, Shape<N>, ...>& shm` mutable reference (D-05 convention, line 216)
- `#pragma unroll` loop pattern (lines 232-233, 296-297)
- `operator()(auto... indices)` for single-index SharedTensor access (line 185)
- `.data[i]` array access for Tensor data (line 87, 232)
- Separate `SharedTLayout` + `TLayout` template params — `TLayout::REG_SIZE` from distributed layout, `SharedTLayout` for shared tensor's byte-offset evaluation (RESEARCH.md A3)

**`SharedTensor::operator()` reference** (lines 185-211):
```cpp
// Returns T& (mutable reference — supports read+write, D-04)
__device__ T& operator()(auto... indices) {
    // Row-major flatten: flatIndex = sum(indices[k] * stride[k])
    // then: logicalOffset = TLayout::evaluate(flatIndex, ...)
    // then: byteOffset = dot(logicalOffset, byteStrides)
    // returns data[byteOffset]
}
```

**`Tensor::data[]` reference** (lines 85-99):
```cpp
template<typename T, typename TShape, typename TLayout>
struct Tensor{
    T data[TLayout::REG_SIZE];  // register-level storage
    // ...
};
```

---

### `python/test/gluon/test_extern_call.py` — ADD 3 test functions (pytest)

**Analog:** `test_elementwise_add` (lines 73-81) + `test_reduce_f16_f32` (lines 116-137)

#### Pattern A: Basic Test Structure (from `test_elementwise_add`, lines 73-81)

**Imports** (lines 1-10) — already present, no new imports needed:
```python
import torch
import pytest
from unittest.mock import patch

import triton
from triton._internal_testing import is_cuda
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

pytestmark = pytest.mark.skipif(not is_cuda(), reason="CUDA-only test")
```

**Test function pattern** (lines 73-81):
```python
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

**Key conventions to replicate:**
- `torch.set_default_device('cuda')` at start of each test function (lines 75, 85, 96, 105, 119)
- `torch.randn(...)` for inputs, `torch.empty_like(...)` for outputs (lines 76-78)
- Kernel launch: `kernel_name[(1,)](...)` with `num_warps=1` (lines 79, 88, 98, 110, 122)
- `torch.cuda.synchronize()` after launch (lines 80, 89, 99, 111, 123)
- `torch.testing.assert_close(out, ref)` for final assertion (lines 81, 91, 100, 113, 127)
- Reference computed entirely in Python/torch — no custom kernels for expected values (D-29)

#### Pattern B: Kernel with Shared Memory + gl.call() + gl.barrier() (from `test_lowerings.py`, lines 576-592)

**Kernel structure for shared memory:**
```python
@gluon.jit
def kernel(x_ptr, y_ptr, shape: ttgl.constexpr, layout: ttgl.constexpr, shared_layout: ttgl.constexpr):
    if len(shape) == 1:
        offs = ttgl.arange(0, shape[0], layout=layout)
    else:
        offs_m = ttgl.arange(0, shape[0], layout=ttgl.SliceLayout(1, layout))[:, None]
        offs_n = ttgl.arange(0, shape[1], layout=ttgl.SliceLayout(0, layout))[None, :]
        offs = offs_m * shape[1] + offs_n
    x = ttgl.load(x_ptr + offs)
    smem = ttgl.allocate_shared_memory(x.dtype, shape, shared_layout)
    smem.store(x)
    y = smem.load(layout)
    ttgl.store(y_ptr + offs, y)
```

**Shared memory API calls (mapped to `gl` namespace in test_extern_call.py):**
- `gl.allocate_shared_memory(gl.float32, [shape...], shared_layout)` → returns `shared_memory_descriptor` (lines 613-630, `_core.py`)
- `shm.load(distributed_layout)` → loads shared memory into a distributed tensor (lines 334-345, `_core.py`)
- `shm.store(tensor)` → stores distributed tensor into shared memory (lines 348-355, `_core.py`)
- `gl.barrier()` → CTA-wide synchronization barrier (lines 703-714, `_core.py`)

**`gl.call()` with void-returning functions:** `gl.call()` in `_core.py` lines 775-813 has `result_layout` as a required parameter. For void-returning functions:
- Test passing `result_layout=[]` (empty list) — the `call()` wrapper decomposes this to `result_layouts=[]`, which `call_extern()` handles (lines 809-810).
- **Fallback if void fails:** wrap `shared_accumulate` in a C++ function returning a 1-element dummy `Tensor<T, Shape<1>, PlaceholderLayout>` and use `result_layout=gl.BlockedLayout([1], [1], [1], [0])` (RESEARCH.md Pitfall #1).

**`SharedLinearLayout` constructor** (from `_layouts.py` lines 630-673):
```python
gl.SharedLinearLayout(offset_bases=[[1,0],[2,0],...,[0,16]], block_bases=[], alignment=16)
```

#### Pattern C: compiled.asm PTX Access (from `test_reduce_f16_f32`, lines 116-137)

**Capturing compiled object + accessing IR:**
```python
compiled = reduce_f16_kernel[(1,)](x, out, num_warps=1)
torch.cuda.synchronize()
# ... assertions ...
ttgir = compiled.asm["ttgir"]
assert "f32" in ttgir
```

**PTX grep pattern for L-01 landmine (D-31):**
```python
compiled = kernel[(1,)](out, ..., num_warps=1)
torch.cuda.synchronize()
torch.testing.assert_close(out, ref)

# D-31: L-01 landmine — verify shared memory addrspace in PTX
ptx = compiled.asm["ptx"]
assert "ld.shared" in ptx or "st.shared" in ptx, (
    f"L-01 LANDMINE: Expected ld.shared or st.shared in PTX but found neither.\n"
    f"AS3 pointer may have been erased through memory. First 200 chars:\n{ptx[:200]}"
)
```

**Note:** `compiled.asm["ptx"]` also used in `test_lowerings.py` line 197-199 for PTX verification.

#### Pattern D: @pytest.mark.parametrize (from `test_elementwise_add`, lines 73-74, and `test_split_add_tuple`, lines 103-104)

```python
@pytest.mark.parametrize("BLOCK", [512])
def test_elementwise_add(BLOCK):
    # ... BLOCK used in test ...
```

For SHTEST-02 (4 swizzle patterns):
```python
@pytest.mark.parametrize("offset_bases,block_bases,label", [
    ([[1,0],[2,0],[4,0],[8,0],[0,1],[0,2],[0,4],[0,8],[0,16]], [], "identity"),
    ([[2,0],[1,0],[4,0],[8,0],[0,1],[0,2],[0,4],[0,8],[0,16]], [], "offset_only"),
    ([[0,1],[2,0],[4,0],[8,0],[1,0],[0,2],[0,4],[0,8],[0,16]], [], "block_only"),
    ([[0,16],[2,0],[0,8],[8,0],[0,4],[4,0],[0,2],[1,0],[0,1]], [], "full_xor"),
])
def test_swizzle_round_trip(offset_bases, block_bases, label):
    ...
```

#### Pattern E: Gluon Kernel with gl.call() + Shared Memory

**Canonical kernel shape for SHTEST-01 sequential test (reuses `process_shared_2d`):**
```python
@gluon.jit
def shared_read_write_kernel(out_ptr, SCALE: gl.constexpr):
    shared_layout: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[[1,0],[2,0],[4,0],[8,0],[0,1],[0,2],[0,4],[0,8],[0,16]],
        block_bases=[], alignment=16
    )
    dist_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 2], [1, 1], [1, 0])

    shm = gl.allocate_shared_memory(gl.float32, [32, 16], shared_layout)
    gl.call("python/test/gluon/tt_plugin.cu", "process_shared_2d", shm, SCALE, result_layout=[])
    gl.barrier()
    result = shm.load(dist_layout)
    gl.store(out_ptr, result)
```

**Canonical kernel shape for SHTEST-01 mixed-args test (uses new `shared_accumulate`):**
```python
@gluon.jit
def shared_accumulate_kernel(x_ptr, out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    shared_layout: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[[1],[2],[4],[8],[16],[32],[64],[128]],
        block_bases=[], alignment=16
    )

    idx = gl.arange(0, 256, layout=layout)
    vals = gl.load(x_ptr + idx)

    shm = gl.allocate_shared_memory(gl.float32, [256], shared_layout)
    gl.call("python/test/gluon/tt_plugin.cu", "shared_accumulate", shm, vals, result_layout=[])
    gl.barrier()
    result = shm.load(layout)
    gl.store(out_ptr + idx, result)
```

**Kernel launch conventions (from existing tests):**
- Always `kernel_name[(1,)](out_ptr_args, ..., num_warps=1)` — single CTA, single warp (lines 79, 88, 98, 122)
- `torch.set_default_device('cuda')` inside test function (line 75)
- `torch.randn(shape, dtype=torch.float32)` for input tensors (lines 76-77)
- `torch.empty(shape, dtype=torch.float32)` or `torch.empty_like(x)` for output tensors (lines 78, 109)
- `torch.cuda.synchronize()` after launch (line 80)
- `torch.testing.assert_close(out, expected)` for verification (line 81)

#### Pattern F: Python-Side Swizzle Reference (SHTEST-02)

**Replicating SharedLinearLayout::evaluate() in Python (from `tt_plugin.cu` lines 160-166):**
```python
def evaluate_shared(flat_index: int, offset_bases):
    """Replicate tt_plugin.cu:160-166 — XOR-add basis rows for set bits."""
    rank = len(offset_bases[0])
    result = [0] * rank
    for bit_pos, basis_row in enumerate(offset_bases):
        if (flat_index >> bit_pos) & 0x1:
            for d in range(rank):
                result[d] ^= basis_row[d]
    return result

def compute_expected(shape, offset_bases, elem_size=4):
    """Compute expected values: each (row,col) maps to byte offset after swizzle."""
    rows, cols = shape
    expected = torch.zeros((rows, cols), dtype=torch.float32)
    for r in range(rows):
        for c in range(cols):
            flat = r * cols + c
            logical = evaluate_shared(flat, offset_bases)
            # Row-major byte offset: sum(logical[d] * stride[d]) * elem_size
            byte_offset = (logical[0] * cols + logical[1]) * elem_size
            expected[r, c] = float(byte_offset)
    return expected
```

---

## Shared Patterns

### Gluon JIT Kernel Pattern
**Source:** `python/test/gluon/test_extern_call.py` lines 13-21
**Apply to:** All 3 new kernel definitions
```python
@gluon.jit
def my_kernel(ptr_args, ..., CONST: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([...], [...], [...], [...])
    # ... kernel body using gl.load/gl.call/gl.barrier/gl.store ...
```
**Key details:**
- `gl.constexpr` on layout and shape parameters
- `gluon.jit` decorator (not `triton.jit` — note the `from triton.experimental import gluon` import)
- Shared memory allocation: `gl.allocate_shared_memory(dtype, shape, layout)` (line 207, `_core.py:613`)
- Barrier: `gl.barrier()` (line 703, `_core.py`)
- Shared load: `shm.load(dist_layout)` returns distributed tensor (line 334, `_core.py`)

### Pytest Test Structure
**Source:** `python/test/gluon/test_extern_call.py` lines 73-81
**Apply to:** All 3 new test functions
```python
def test_name():
    torch.set_default_device('cuda')
    x = torch.randn((shape,), dtype=torch.float32)
    out = torch.empty_like(x)
    kernel[(1,)](x, out, num_warps=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected)
```

### PTX Access + L-01 Landmine Guard
**Source:** `python/test/gluon/test_extern_call.py` line 122 + `test_lowerings.py` lines 197-201
**Apply to:** SHTEST-01 both tests + SHTEST-02 all 4 parametrized patterns
```python
compiled = kernel[(1,)](...)
ptx = compiled.asm["ptx"]
assert "ld.shared" in ptx or "st.shared" in ptx
```

### API Surface Summary (used across all kernels)

| Operation | Python API | Source |
|-----------|-----------|--------|
| Shared memory allocation | `gl.allocate_shared_memory(gl.float32, shape, shared_layout)` | `_core.py:613` |
| Barrier | `gl.barrier()` | `_core.py:703` |
| Shared→distributed load | `shm.load(distributed_layout)` | `_core.py:334` |
| Distributed→shared store | `shm.store(tensor)` | `_core.py:348` |
| Extern call | `gl.call("tt_plugin.cu", "func", args, result_layout=layout)` | `_core.py:775` |
| Extern call (void) | `gl.call("tt_plugin.cu", "func", args, result_layout=[])` | `_core.py:775` + `_semantic.py:250` |

## Unresolved: Void-Returning gl.call()

**Issue:** All 6 existing `test_extern_call.py` tests call tensor-returning functions. `process_shared_2d` and `shared_accumulate` are `__device__ void`. The `gl.call()` function in `_core.py:775` has `result_layout` as a required parameter; passing `result_layout=[]` should decompose to `result_layouts=[]` (line 810) which `call_extern()` handles with zero result types (lines 303-337 of `_semantic.py`).

**Verification step (before writing tests):** Test minimal void call first:
```python
# Quick smoke test in a throwaway script:
@gluon.jit
def smoke_kernel(out_ptr):
    shm = gl.allocate_shared_memory(gl.float32, [256], shared_layout)
    gl.call("python/test/gluon/tt_plugin.cu", "write_shared_1d", shm, 1.0, result_layout=[])
```

**Fallback if void fails:** Replace void calls with C++ wrapper functions that return a 1-element dummy tensor. Example:
```cpp
// In tt_plugin.cu — fallback wrapper if void return unsupported:
template<typename T, typename TLayout>
__device__ Tensor<T, Shape<1>, PlaceholderLayout> wrapper_process_shared_2d(
    SharedTensor<T, Shape<32, 16>, TLayout>& shm, T scale)
{
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 16; j++)
            shm(i, j) = shm(i, j) * scale;
    Tensor<T, Shape<1>, PlaceholderLayout> dummy;
    dummy.data[0] = T{};
    return dummy;
}
```
Then call: `gl.call(..., result_layout=gl.BlockedLayout([1],[1],[1],[0]))`.

## No Analog Found

All files have close analog matches in the codebase. No files require fallback to RESEARCH.md-only patterns.

---

## Metadata

**Analog search scope:** `python/test/gluon/`, `python/triton/experimental/gluon/language/`
**Files scanned:** 8 (test_extern_call.py, tt_plugin.cu, test_lowerings.py, test_shared_tensor.py, _core.py, _layouts.py, _semantic.py, __init__.py)
**Pattern extraction date:** 2026-07-21
