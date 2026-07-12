# Phase 03: Verification - Pattern Map

**Mapped:** 2026-07-12
**Files analyzed:** 2
**Analogs found:** 2 / 2

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `python/test/gluon/tt_plugin.cu` (append) | device-library | transform | `python/test/gluon/tt_plugin.cu:142-151` (`reduce` fn + `TArg`/`TRes` aliases at 138-139) | exact (same file, same pattern) |
| `python/test/gluon/test_extern_call.py` (append) | test (E2E) | request-response | `python/test/gluon/test_extern_call.py:35-44,82-88` (`reduce_kernel` + `test_reduce_different_shape`) | exact (same file, near-identical structure) |

---

## Pattern Assignments

### `python/test/gluon/tt_plugin.cu` — append new device function (device-library, transform)

**Analog:** `reduce` function lines 138-151

This file is a CUDA C++ device library. All device functions are templated `__device__` returning `Tensor<T, Shape<...>, Layout>` structs. The `TArg`/`TRes` layout aliases on lines 138-139 are reused; the new function mirrors `reduce` structurally but with different element types (half in, float out) and an f32 accumulator.

**Layout aliases (lines 138-139) — reuse directly:**
```cpp
using TArg = typename TensorLayout<Shape<32, 32>, 1>::Layout<{{0,1},{0,2},{0,4},{0,8},{0,16}},{{1,0},{2,0},{4,0},{8,0},{16,0}},{}>;
using TRes = typename TensorLayout<Shape<32>, 1>::Layout<{},{{1},{2},{4},{8},{16}},{}>;
```

**The `reduce` function (lines 142-151) — adapt for f16→f32:**
```cpp
template<typename T>
__device__ Tensor<T, Shape<32>, TRes> reduce(const Tensor<T, Shape<32, 32>, TArg>& Vals){
    Tensor<T, Shape<32>, TRes>  Result;
    Result.data[0] = T{};
    #pragma unroll
    for(int i = 0; i < 32; i++){
        Result.data[0] += Vals.data[i];
    }
    return Result;
}
```

**Adaptation recipe for the new `reduce_f16` (or equivalent name):**
1. Copy the `reduce` function structure verbatim.
2. Change the input element type from `T` to `half` (or explicit `__half`): `const Tensor<half, Shape<32, 32>, TArg>& Vals`.
3. Change the return element type from `T` to `float`: `Tensor<float, Shape<32>, TRes>`.
4. Change the accumulator initialization from `Result.data[0] = T{}` to `Result.data[0] = float{}` (or `0.0f`).
5. Keep the `#pragma unroll` / `for` loop identical — `half` → `float` promotion is implicit in `+=`.
6. Place the new function after the existing `reduce` (after line 151), before `split_add`.

**Concrete new code pattern:**
```cpp
// f16 -> f32 reduction: same layout structure as reduce, different element types
__device__ Tensor<float, Shape<32>, TRes> reduce_f16(const Tensor<half, Shape<32, 32>, TArg>& Vals){
    Tensor<float, Shape<32>, TRes>  Result;
    Result.data[0] = float{};
    #pragma unroll
    for(int i = 0; i < 32; i++){
        Result.data[0] += Vals.data[i];
    }
    return Result;
}
```

**Key structural constraints:**
- The `PlaceholderLayout` probe (lines 81-99) deduces return type from the **declared return type** of the template function — so `Tensor<float, Shape<32>, TRes>` is what the inference reads.
- `TArg` and `TRes` already encode the `Shape<32,32>` → `Shape<32>` transition; no new aliases needed.
- No `#include` changes required — `half` is a built-in CUDA type.

---

### `python/test/gluon/test_extern_call.py` — append new kernel + test (test/E2E, request-response)

**Analog 1:** `reduce_kernel` (lines 35-44) — the `@gluon.jit` kernel pattern
**Analog 2:** `test_reduce_different_shape` (lines 82-88) — the pytest test function pattern
**Analog 3:** Module-level setup (lines 1-10) — imports + pytestmark (no changes needed here)

#### Imports / module-level setup (lines 1-10 — unchanged, for reference)

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

No import changes are needed. The existing imports cover `torch`, `pytest`, `gluon`, `gl`, and `is_cuda`.

#### Kernel pattern (`reduce_kernel`, lines 35-44) — adapt for f16→f32

```python
@gluon.jit
def reduce_kernel(x_ptr, out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1, 32], [32, 1], [1, 1], [1, 0])
    idx = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))[None, :] \
        + 32 * gl.arange(0, 32, layout=gl.SliceLayout(1, layout))[:, None]
    x_vals = gl.load(x_ptr + idx)
    out_idx = gl.arange(0, 32, layout=gl.SliceLayout(1, layout))
    red_vals = gl.call("python/test/gluon/tt_plugin.cu", "reduce", x_vals,
                        result_layout=gl.SliceLayout(1, layout))
    gl.store(out_ptr + out_idx, red_vals)
```

**Adaptation recipe for the new `reduce_f16_kernel`:**
1. Copy the kernel structure exactly (same layout, same indexing, same 2D→1D slice pattern).
2. Change the function name from `"reduce"` to `"reduce_f16"` (or the chosen device function name).
3. No other changes — the kernel itself is dtype-agnostic; the f16→f32 transition is in the `.cu` device function, not the kernel.

**Concrete new kernel pattern:**
```python
@gluon.jit
def reduce_f16_kernel(x_ptr, out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1, 32], [32, 1], [1, 1], [1, 0])
    idx = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))[None, :] \
        + 32 * gl.arange(0, 32, layout=gl.SliceLayout(1, layout))[:, None]
    x_vals = gl.load(x_ptr + idx)
    out_idx = gl.arange(0, 32, layout=gl.SliceLayout(1, layout))
    red_vals = gl.call("python/test/gluon/tt_plugin.cu", "reduce_f16", x_vals,
                        result_layout=gl.SliceLayout(1, layout))
    gl.store(out_ptr + out_idx, red_vals)
```

#### Test pattern (`test_reduce_different_shape`, lines 82-88) — adapt for f16→f32 + assertions

```python
def test_reduce_different_shape():
    torch.set_default_device('cuda')
    x = torch.randn((32, 32), dtype=torch.float32)
    out = torch.empty((32,), dtype=torch.float32)
    reduce_kernel[(1,)](x, out, num_warps=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x.sum(1))
```

**Adaptation recipe for `test_reduce_f16_f32`:**
1. Input `x` is f16: `torch.randn((32, 32), dtype=torch.float16)`.
2. Output `out` is f32: `torch.empty((32,), dtype=torch.float32)`.
3. Torch reference uses f32 accumulation: `x.to(torch.float32).sum(1)`.
4. Add `rtol=1e-2, atol=1e-2` tolerance to `assert_close` (f16 input requires relaxed tolerance).
5. Add an **inferred-type assertion** on `reduce_f16_kernel.asm['ttgir']` (D-05): the compiled kernel's `ttg.extern_call` result type must contain `f32` element type and `32` shape.

**Concrete new test pattern:**
```python
def test_reduce_f16_f32():
    """
    f16 -> f32 reduction: verifies shape AND dtype are inferred from CUDA side.
    Only result_layout is supplied; f32 element type + [32] shape come from inference.
    """
    torch.set_default_device('cuda')
    x = torch.randn((32, 32), dtype=torch.float16)
    out = torch.empty((32,), dtype=torch.float32)
    reduce_f16_kernel[(1,)](x, out, num_warps=1)
    torch.cuda.synchronize()

    # Numeric check: f16 input, f32 accumulation, relaxed tolerance
    ref = x.to(torch.float32).sum(1)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    # Inferred-type assertion: prove inference produced f32 + [32] before lowering.
    # The ttg.extern_call result type encodes the inferred element type + shape.
    ttgir = reduce_f16_kernel.asm["ttgir"]
    assert "f32" in ttgir, (
        f"Expected f32 element type in ttg.extern_call result, but got:\n{ttgir}"
    )
    # The result tensor type should carry shape [32] (reduced from [32,32]).
    # Look for tensor<32xf32 — the 32-element dimension.
    assert "tensor<32xf32" in ttgir, (
        f"Expected tensor<32xf32 result shape in ttg.extern_call, but got:\n{ttgir}"
    )
```

#### Kernel dispatch pattern (call site) — from all existing tests (lines 62-101)

Every test follows this exact pattern:
```python
torch.set_default_device('cuda')                      # set default device
x = torch.randn(SHAPE, dtype=DTYPE)                    # build inputs
out = torch.empty(OUT_SHAPE, dtype=OUT_DTYPE)          # build output buffer
kernel[(1,)](x, out, num_warps=1)                      # launch kernel
torch.cuda.synchronize()                               # wait for GPU
torch.testing.assert_close(out, ref, ...)              # compare
```

The `kernel.asm["ttgir"]` property is available on the `GluonJITFunction` (which extends `triton.JITFunction`) **after** the first invocation triggers compilation. Access it after `kernel[(1,)](...)` returns.

#### Assert on `kernel.asm` — cross-test reference

From `python/test/unit/language/test_core.py:3122` and `python/test/unit/plugins/test_plugin.py:41-54`:

```python
# Count-based assertion (test_core.py:3122):
assert compiled_kernel.asm["ttgir"].count('"tt.reduce"') == 1, "we shouldn't rematerialize tt.reduce"

# Contains-based assertion (test_plugin.py:41):
assert "tt.func public @foo" not in h.asm["ttgir"]

# Contains-based assertion (test_plugin.py:45):
assert "tt.func public @foo" in h.asm["ttgir"]
```

The recommended pattern for D-05 uses `in` membership test on the ttgir string, with a descriptive assertion message showing the failing IR for debugging.

---

## Shared Patterns

### Test function structure
**Source:** `test_extern_call.py` lines 61-101 (all 5 existing tests)
**Apply to:** `test_reduce_f16_f32`

```python
def test_NAME():
    torch.set_default_device('cuda')
    x = torch.randn((...), dtype=TORCH_DTYPE)            # input tensor
    out = torch.empty((...), dtype=TORCH_DTYPE)           # output buffer
    KERNEL[(1,)](x, out, num_warps=1)                     # launch
    torch.cuda.synchronize()                              # sync
    torch.testing.assert_close(out, REF)                  # verify
```

### `gl.call()` signature
**Source:** `test_extern_call.py` lines 19-20, 29-31, 42-43, 53-56, 113-115
**Apply to:** `reduce_f16_kernel`

```python
gl.call("python/test/gluon/tt_plugin.cu", "FUNC_NAME", INPUT_TENSOR,
        result_layout=LAYOUT)
```

### Device function structure
**Source:** `tt_plugin.cu` lines 104-110, 115-131, 142-151
**Apply to:** the new `reduce_f16` device function

```cpp
template</* ... */>
__device__ Tensor<RET_T, Shape<RET_DIMS...>, RET_LAYOUT> FUNC_NAME(
    const Tensor<ARG_T, Shape<ARG_DIMS...>, ARG_LAYOUT>& ARG)
{
    Tensor<RET_T, Shape<RET_DIMS...>, RET_LAYOUT> Result;
    Result.data[0] = RET_T{};
    #pragma unroll
    for(int i = 0; i < N; i++){
        Result.data[0] += ARG.data[i];
    }
    return Result;
}
```

### CUDA-side error handling
**Source:** `tt_plugin.cu` lines 92-98 (the `Tensor` converting constructor)
**Apply to:** No error handling needed — the existing `static_assert` in the `PlaceholderLayout` converter protects against dtype/shape mismatch at compile time. No runtime error paths in device functions.

### MLIR assertion pattern (for D-05)
**Source:** `test_core.py:3122`, `test_plugin.py:41-54`
**Apply to:** `test_reduce_f16_f32`

```python
ttgir = kernel.asm["ttgir"]
assert "EXPECTED_TEXT" in ttgir, f"message showing failing IR:\n{ttgir}"
```

---

## Key Constraints from CONTEXT.md

| Decision | Constraint |
|----------|------------|
| D-01 | Test exercises **shape AND dtype together** — only one new test |
| D-02 | New device fn: `Tensor<half, Shape<32,32>, TArg>` → `Tensor<float, Shape<32>, TRes>` |
| D-03 | Torch ref: `x.to(torch.float32).sum(1)` |
| D-04 | Kernel passes `result_layout=gl.SliceLayout(1, layout)` **only** — no hand-computed shape/dtype |
| D-05 | Assert inferred result type via `kernel.asm['ttgir']` — confirm f32 + [32] |
| D-06 | `make test-lit` is a separate regression gate (not in the test itself) |
| D-07 | `rtol=1e-2, atol=1e-2` tolerance for f16→f32 accumulation |
| D-08 | Only f16→f32 tested this phase; bf16/i32 deferred |

---

## No Analog Found

All files have exact analogs within the same files they modify. No searches needed beyond the target files themselves.

---

## Metadata

**Analog search scope:** `python/test/gluon/test_extern_call.py`, `python/test/gluon/tt_plugin.cu`, `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` (for MLIR format verification), `python/triton/experimental/gluon/language/_semantic.py` (for call_extern signature), `python/triton/experimental/gluon/_runtime.py` (for GluonJITFunction .asm inheritance)
**Files scanned:** 7 (2 primary analogs + 5 supporting references)
**Pattern extraction date:** 2026-07-12
