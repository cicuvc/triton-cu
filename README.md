# triton-cu

**Triton with in-process CUDA C++ interop via `gl.call()`**

A fork of [Triton](https://github.com/triton-lang/triton) that adds first-class CUDA C++ extern function support. The `gl.call()` mechanism allows Triton/Gluon kernels to invoke external CUDA C++ template functions compiled in-process via clang, enabling fine-grained control and semantics that are difficult to express in pure Triton DSL.

## Motivation

Triton excels at expressing tensor computations declaratively, but certain patterns — custom reduction algorithms, warp-level communication, non-standard data layouts, and intricate control flow — can be awkward or impossible to express in the DSL. CUDA C++ provides full expressiveness for these cases, but historically bridging Triton and CUDA required:

1. Writing and precompiling CUDA wrappers in a separate build step
2. Passing `extern_libs` as bitcode paths at kernel launch
3. Dealing with `extern "C"` calling conventions that strip rich type information

**triton-cu** eliminates this friction by integrating clang's CodeGen directly into Triton's compiler pipeline, enabling:

- **Template instantiation at JIT time** — write generic CUDA C++ template functions; clang instantiates them with concrete tensor shapes/layouts during compilation
- **Rich type system** — pass `Tensor<T, Shape, Layout>` objects directly to CUDA functions with full type information preserved
- **Zero manual compilation** — no separate `nvcc` step, no `.bc` file management
- **Seamless inlining** — extern functions are inlined via `alwaysinline` + O3, producing optimized LLVM IR indistinguishable from hand-written Triton code

## Architecture

```
Triton/Gluon kernel with gl.call()
        │
        ▼
  MLIR tt.call_external op
  (extern_elementwise)
        │
        ▼
  Extract extern call specs ───► CUDACompiler (clang CodeGen)
  (symbol + operand types)      instantiate C++ template functions
        │                                  │
        │                         bitcode bytes (LLVM Module)
        │                                  │
        ▼                                  │
  ttgir → llir lowering                    │
  (ExternCallOpToLLVM:                      │
   { [N x scalar] } return type,           │
   by-pointer arg passing)                  │
        │                                  │
        ▼                                  ▼
  llvm.to_module ◄── CloneFunctionInto (same LLVMContext)
  (link_cuda_bitcode)   + ret-type fix + alwaysinline
        │
        ▼
  LLVM O3 optimization (full inlining)
        │
        ▼
  PTX → cubin
```

### Key Components

| Stage | File | Role |
|-------|------|------|
| **Python pipeline** | `third_party/nvidia/backend/compiler.py` | Pre-compile extern calls, link bitcode |
| **clang compiler** | `python/src/clang_compiler.cc` / `.h` | CUDACompiler, TensorTypeHelpers, CustomAstConsumer |
| **Python bindings** | `python/src/llvm.cc` | `extract_extern_call_specs`, `compile_cuda_to_module`, `link_cuda_bitcode` |
| **MLIR→LLVM lowering** | `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` | clang-compatible struct types |
| **Device library** | `tt_plugin.cu` | Template tensor primitives (Shape, TensorLayout, Layout, Tensor) |

## Example

```python
import triton
import triton.language as tl
import gluon

@triton.jit
def vectorized_add(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, N))
    y = tl.load(y_ptr + tl.arange(0, N))
    # gl.call: invoke CUDA C++ template function in-process
    out = gluon.call("elementwise_add", x, y)
    tl.store(out_ptr + tl.arange(0, N), out)
```

Where `elementwise_add` is defined as a C++ template:

```cpp
template<typename T, uint32_t N, typename Layout>
__host__ __device__
Tensor<T, Shape<N>, Layout> elementwise_add(
    const Tensor<T, Shape<N>, Layout>& lhs,
    const Tensor<T, Shape<N>, Layout>& rhs)
{
    Tensor<T, Shape<N>, Layout> result;
    #pragma unroll
    for (uint32_t i = 0; i < N; i++)
        result[i] = lhs[i] + rhs[i];
    return result;
}
```

The compiler instantiates `elementwise_add<float, 512, ConcreteLayout>` at JIT time using clang, links the bitcode into the kernel, and O3 fully inlines the function.

## Build

```bash
export LLVM_SYSPATH=/path/to/llvm/install
cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DTRITON_BUILD_PYTHON_MODULE=ON \
  -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
  -DLLVM_SYSPATH=${LLVM_SYSPATH} \
  -B build .
ninja -C build triton
```

Requires a clang-enabled LLVM build (LLVM + clang + lld).

## Related

- [Triton](https://github.com/triton-lang/triton) — upstream project
- [Gluon](https://github.com/triton-lang/triton) — Triton's composable kernel frontend
- [NKS Lab](https://github.com/.../nks) — reference cu_compiler implementation
