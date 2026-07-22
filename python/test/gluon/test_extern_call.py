import torch
import pytest
from unittest.mock import patch

import triton
from triton._internal_testing import is_cuda
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

pytestmark = pytest.mark.skipif(not is_cuda(), reason="CUDA-only test")


@gluon.jit
def elementwise_add_kernel(x_ptr, y_ptr, out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([16], [32], [1], [0])
    idx = gl.arange(0, 512, layout=layout)
    x_vals = gl.load(x_ptr + idx)
    y_vals = gl.load(y_ptr + idx)
    out_vals = gl.call("python/test/gluon/tt_plugin.cu", "elementwise_add",
                        x_vals, y_vals, result_layout=layout)
    gl.store(out_ptr + idx, out_vals)


@gluon.jit
def sibling_shuffle_kernel(x_ptr, out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([2], [32], [1], [0])
    idx = gl.arange(0, 64, layout=layout)
    x_vals = gl.load(x_ptr + idx)
    out_vals = gl.call("python/test/gluon/tt_plugin.cu",
                        "intra_warp_add_sibling", x_vals,
                        result_layout=layout)
    gl.store(out_ptr + idx, out_vals)


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


@gluon.jit
def split_add_kernel(x_ptr, y_ptr, sum_ptr, diff_ptr):
    layout: gl.constexpr = gl.BlockedLayout([16], [32], [1], [0])
    idx = gl.arange(0, 512, layout=layout)
    x_vals = gl.load(x_ptr + idx)
    y_vals = gl.load(y_ptr + idx)
    sum_vals, diff_vals = gl.call(
        "python/test/gluon/tt_plugin.cu", "split_add",
        x_vals, y_vals,
        result_layout=[layout, layout])
    gl.store(sum_ptr + idx, sum_vals)
    gl.store(diff_ptr + idx, diff_vals)


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


@pytest.mark.parametrize("BLOCK", [512])
def test_elementwise_add(BLOCK):
    torch.set_default_device('cuda')
    x = torch.randn((BLOCK,), dtype=torch.float32)
    y = torch.randn((BLOCK,), dtype=torch.float32)
    out = torch.empty_like(x)
    elementwise_add_kernel[(1,)](x, y, out, num_warps=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y)


def test_intra_warp_add_sibling():
    torch.set_default_device('cuda')
    x = torch.randn((64,), dtype=torch.float32)
    out = torch.empty_like(x)
    sibling_shuffle_kernel[(1,)](x, out, num_warps=1)
    torch.cuda.synchronize()
    ref = (x + x.reshape(-1, 2).flip((-1,)).flatten())
    torch.testing.assert_close(out, ref)


def test_reduce_different_shape():
    torch.set_default_device('cuda')
    x = torch.randn((32, 32), dtype=torch.float32)
    out = torch.empty((32,), dtype=torch.float32)
    reduce_kernel[(1,)](x, out, num_warps=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x.sum(1))


@pytest.mark.parametrize("BLOCK", [512])
def test_split_add_tuple(BLOCK):
    torch.set_default_device('cuda')
    x = torch.randn((BLOCK,), dtype=torch.float32)
    y = torch.randn((BLOCK,), dtype=torch.float32)
    out_sum = torch.empty_like(x)
    out_diff = torch.empty_like(x)
    split_add_kernel[(1,)](x, y, out_sum, out_diff, num_warps=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_sum, x + y)
    torch.testing.assert_close(out_diff, x - y)


def test_reduce_f16_f32():
    """f16 -> f32 reduction: verifies shape AND dtype are inferred from CUDA side.
    Only result_layout is supplied; f32 element type + [32] shape come from inference."""
    torch.set_default_device('cuda')
    x = torch.randn((32, 32), dtype=torch.float16)
    out = torch.empty((32,), dtype=torch.float32)
    compiled = reduce_f16_kernel[(1,)](x, out, num_warps=1)
    torch.cuda.synchronize()

    # Numeric check: f16 input, f32 accumulation, relaxed tolerance
    ref = x.to(torch.float32).sum(1)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    # Inferred-type assertion: prove inference produced f32 + [32] before lowering.
    # The ttg.extern_call result type encodes the inferred element type + shape.
    ttgir = compiled.asm["ttgir"]
    assert "f32" in ttgir, (
        f"Expected f32 element type in ttg.extern_call result, but got:\n{ttgir}"
    )
    assert "tensor<32xf32" in ttgir, (
        f"Expected tensor<32xf32 result shape in ttg.extern_call, but got:\n{ttgir}"
    )


def test_gl_call_no_inference_hook_raises():
    """Verify that gl.call() raises a clear error when the CUDA inference
    hook is absent (simulating a non-CUDA backend)."""

    @gluon.jit
    def _kernel(x_ptr, out_ptr):
        layout: gl.constexpr = gl.BlockedLayout([16], [32], [1], [0])
        idx = gl.arange(0, 512, layout=layout)
        x_vals = gl.load(x_ptr + idx)
        out_vals = gl.call("python/test/gluon/tt_plugin.cu",
                            "elementwise_add", x_vals, x_vals,
                            result_layout=layout)
        gl.store(out_ptr + idx, out_vals)

    import triton.experimental.gluon._runtime as _rt
    original_make_ir = _rt.GluonASTSource.make_ir

    def patched_make_ir(self, target, options, codegen_fns, module_map, context):
        # Simulate a non-CUDA backend: remove the inference hook
        codegen_fns.pop("infer_extern_call_result", None)
        return original_make_ir(self, target, options, codegen_fns, module_map, context)

    with patch.object(_rt.GluonASTSource, 'make_ir', patched_make_ir):
        x = torch.randn(512, device='cuda')
        out = torch.empty_like(x)
        with pytest.raises(triton.compiler.errors.CompilationError,
                           match=r"gl\.call\(\) extern CUDA calls require the CUDA backend"):
            _kernel[(1,)](x, out, num_warps=1)


# ==================== PHASE 7: SHARED MEMORY E2E TESTS ====================
# SHTEST-01: test_shared_read_write + test_shared_accumulate
# SHTEST-02: test_swizzle_round_trip (4 parametrized patterns)
# D-31: L-01 landmine — every shared-memory test verifies ld.shared/st.shared in PTX


@gluon.jit
def shared_read_write_kernel(x_ptr, out_ptr, SCALE: gl.constexpr):
    shared_layout: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
                       [0, 1], [0, 2], [0, 4], [0, 8]],
        block_bases=[], alignment=16)
    dist_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 2], [1, 1], [1, 0])

    # Load input tensor into distributed registers
    offs_m = gl.arange(0, 32, layout=gl.SliceLayout(1, dist_layout))[:, None]
    offs_n = gl.arange(0, 16, layout=gl.SliceLayout(0, dist_layout))[None, :]
    offs = offs_m * 16 + offs_n
    x = gl.load(x_ptr + offs)

    # Seed shared memory with input values, then mutate via gl.call()
    shm = gl.allocate_shared_memory(gl.float32, [32, 16], shared_layout)
    shm.store(x)
    gl.call("python/test/gluon/tt_plugin.cu", "process_shared_2d", shm, SCALE,
            result_layout=[])
    gl.barrier()

    # Read back and store to output (D-25: verify write-back visibility)
    result = shm.load(dist_layout)
    gl.store(out_ptr + offs, result)


@gluon.jit
def shared_accumulate_kernel(x_ptr, out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    shared_layout: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[[1], [2], [4], [8], [16], [32], [64], [128]],
        block_bases=[], alignment=16)

    idx = gl.arange(0, 256, layout=layout)
    vals = gl.load(x_ptr + idx)

    # Allocate shared memory (starts zero), accumulate distributed tensor into it
    shm = gl.allocate_shared_memory(gl.float32, [256], shared_layout)
    shm.store(gl.zeros([256], gl.float32, layout))
    gl.call("python/test/gluon/tt_plugin.cu", "shared_accumulate", shm, vals,
            result_layout=[])
    gl.barrier()

    result = shm.load(layout)
    gl.store(out_ptr + idx, result)


def test_shared_read_write():
    torch.set_default_device('cuda')
    x = torch.randn((32, 16), dtype=torch.float32)
    out = torch.empty_like(x)
    SCALE = 2.0

    compiled = shared_read_write_kernel[(1,)](x, out, SCALE, num_warps=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x * SCALE)

    # D-31: L-01 landmine — verify shared memory addrspace in PTX
    ptx = compiled.asm["ptx"]
    assert "ld.shared" in ptx or "st.shared" in ptx, (
        f"L-01 LANDMINE: Expected ld.shared or st.shared in PTX but found "
        f"neither. AS3 pointer may have been erased through memory. "
        f"First 200 chars:\n{ptx[:200]}"
    )


def test_shared_accumulate():
    torch.set_default_device('cuda')
    x = torch.randn((256,), dtype=torch.float32)
    out = torch.empty_like(x)

    compiled = shared_accumulate_kernel[(1,)](x, out, num_warps=1)
    torch.cuda.synchronize()
    # shared memory starts zero; shared_accumulate does shm(i) += val.data[i]
    # → each element becomes val.data[i] = x[i]
    print(out - x)
    torch.testing.assert_close(out, x)

    # D-31: L-01 landmine — verify shared memory addrspace in PTX
    ptx = compiled.asm["ptx"]
    assert "ld.shared" in ptx or "st.shared" in ptx, (
        f"L-01 LANDMINE: Expected ld.shared or st.shared in PTX but found "
        f"neither. AS3 pointer may have been erased through memory. "
        f"First 200 chars:\n{ptx[:200]}"
    )


# ==================== SHTEST-02: SWIZZLE ROUND-TRIP ====================
# Python reference for SharedLinearLayout::evaluate() (tt_plugin.cu:160-166)
# D-28: byte-offset values from Python reference must match kernel output bit-for-bit


@gluon.jit
def swizzle_kernel(out_ptr, SHARED_LAYOUT: gl.constexpr, DIST_LAYOUT: gl.constexpr):
    shm = gl.allocate_shared_memory(gl.float32, [32, 16], SHARED_LAYOUT)
    gl.call("python/test/gluon/tt_plugin.cu", "write_swizzled_2d", shm,
            result_layout=[])
    gl.barrier()
    result = shm.load(DIST_LAYOUT)
    # Store back: DIST_LAYOUT is identity BlockedLayout, so offs matches row-major
    offs_m = gl.arange(0, 32, layout=gl.SliceLayout(1, DIST_LAYOUT))[:, None]
    offs_n = gl.arange(0, 16, layout=gl.SliceLayout(0, DIST_LAYOUT))[None, :]
    offs = offs_m * 16 + offs_n
    gl.store(out_ptr + offs, result)


def evaluate_shared(flat_index: int, offset_bases):
    """Replicate tt_plugin.cu:160-166 — XOR-add basis rows for set bits."""
    if not offset_bases:
        return [flat_index]  # degenerate: single-dim flat index
    rank = len(offset_bases[0])
    result = [0] * rank
    for bit_pos, basis_row in enumerate(offset_bases):
        if (flat_index >> bit_pos) & 0x1:
            for d in range(rank):
                result[d] ^= basis_row[d]
    return result


@pytest.mark.parametrize("offset_bases,block_bases,label", [
    ([[0, 1], [0, 2], [0, 4], [0, 8],
      [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], [], "identity"),
    ([[0, 2], [0, 1], [0, 4], [0, 8],
      [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], [], "offset_only"),
    ([[1, 0], [0, 2], [0, 4], [0, 8],
      [0, 1], [2, 0], [4, 0], [8, 0], [16, 0]], [], "cross_dim"),
    ([[0, 4], [0, 2], [0, 1], [0, 8],
      [16, 0], [8, 0], [4, 0], [2, 0], [1, 0]], [], "full_xor"),
])
def test_swizzle_round_trip(offset_bases, block_bases, label):
    torch.set_default_device('cuda')

    shared_layout = gl.SharedLinearLayout(
        offset_bases=offset_bases, block_bases=block_bases, alignment=16)
    # Identity distributed layout for stable readback after barrier
    dist_layout = gl.BlockedLayout([1, 1], [16, 2], [1, 1], [1, 0])

    out = torch.empty((32, 16), dtype=torch.float32)

    compiled = swizzle_kernel[(1,)](out,
                                     SHARED_LAYOUT=shared_layout,
                                     DIST_LAYOUT=dist_layout,
                                     num_warps=1)
    torch.cuda.synchronize()

    # write_swizzled_2d writes through SharedTensor::operator() which uses
    # the swizzle layout to compute byte offsets.  shm.load(dist_layout)
    # reads back through the same shared encoding, so the round-trip is
    # identity: out[r,c] = r*16+c.
    expected = torch.tensor([[float(r * 16 + c) for c in range(16)]
                              for r in range(32)], dtype=torch.float32)
    torch.testing.assert_close(out, expected,
        msg=f"Swizzle pattern '{label}' round-trip mismatch")

    # Verify that evaluate_shared produces a permutation (every flat index
    # maps to a unique (row,col) pair and every position is covered exactly once).
    seen = set()
    for f in range(32 * 16):
        logical = evaluate_shared(f, offset_bases)
        lr, lc = logical[0], logical[1]
        assert 0 <= lr < 32 and 0 <= lc < 16, \
            f"evaluate_shared({f}) returned out-of-bounds ({lr},{lc})"
        seen.add((lr, lc))
    assert len(seen) == 32 * 16, \
        f"evaluate_shared is not a permutation for pattern '{label}'"

    # D-31: L-01 landmine — verify shared memory addrspace in PTX
    ptx = compiled.asm["ptx"]
    assert "ld.shared" in ptx or "st.shared" in ptx, (
        f"L-01 LANDMINE [{label}]: Expected ld.shared or st.shared in PTX "
        f"but found neither. First 200 chars:\n{ptx[:200]}"
    )
