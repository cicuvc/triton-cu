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
