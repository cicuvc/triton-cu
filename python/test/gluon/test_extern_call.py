import torch
import pytest

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
