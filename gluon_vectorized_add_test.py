import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl

@gluon.jit
def vectorized_add(x_ptr, y_ptr, out_ptr):
    # works with num_warps = 1 for test
    layout: gl.constexpr = gl.BlockedLayout([16], [32], [1], [0]) # 512 eleemnts
    idx = gl.arange(0, 512, layout = layout)

    x_vals = gl.load(x_ptr + idx)
    y_vals = gl.load(y_ptr + idx)

    out_vals = gl.call("tt_plugin.cu", "elementwise_add", x_vals, y_vals)
    # out_vals = x_vals + y_vals

    gl.store(out_ptr + idx, out_vals)

if __name__ == "__main__":
    torch.set_default_device('cuda')

    x = torch.randn((512,), dtype = torch.float32)
    y = torch.randn((512,), dtype = torch.float32)
    out = torch.empty_like(x)

    vectorized_add[(1,)](x, y, out, num_warps = 1)

    torch.cuda.synchronize()

    torch.testing.assert_close(out, x + y)
    print("Test passed")