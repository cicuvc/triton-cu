from pathlib import Path
import sys
sys.path.append('python')
import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl

@gluon.jit
def vectorized_red(x_ptr, out_ptr):
    # works with num_warps = 1 for test
    layout: gl.constexpr = gl.BlockedLayout([1,32], [32,1], [1,1], [1,0]) # 512 eleemnts
    idx = gl.arange(0, 32, layout = gl.SliceLayout(0,layout))[None,:] + 32 * gl.arange(0, 32, layout = gl.SliceLayout(1,layout))[:,None]

    x_vals = gl.load(x_ptr + idx)
    
    out_idx = gl.arange(0, 32, layout = gl.SliceLayout(1,layout))
    red_vals = gl.call("tt_plugin.cu", "reduce", x_vals, result_layout=gl.SliceLayout(1,layout))
    #red_vals = gl.sum(x_vals, 1)
    gl.store(out_ptr + out_idx, red_vals)

if __name__ == "__main__":
    torch.set_default_device('cuda')

    x = torch.randn((32,32), dtype = torch.float32)
    out = torch.empty((32, 1), dtype = torch.float32)

    k = vectorized_red[(1,)](x, out, num_warps=1)

    torch.cuda.synchronize()

    torch.testing.assert_close(out, x.sum(1, True))
