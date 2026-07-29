"""GPU-free tests for the forbidden-synchronization guard in gl.call.

Device functions consumed by gl.call are inlined into the Triton kernel;
CUDA synchronization/fence primitives (__syncthreads, __barrier_sync,
__threadfence*) inside them cannot be modeled by MLIR (and can deadlock
under warp specialization). The in-process CUDA compiler must reject them
at instantiation time, including when reached through same-TU callees.
"""

import glob
import os

import pytest

from triton._C.libtriton import llvm

_LLVM_INSTALL = os.environ.get("LLVM_SYSPATH", "")
if _LLVM_INSTALL:
    _clang_ver = os.path.join(_LLVM_INSTALL, "lib", "clang")
    if os.path.isdir(_clang_ver):
        _vers = sorted(os.listdir(_clang_ver))
        LLVM_RESOURCE_DIR = os.path.join(_clang_ver, _vers[-1]) if _vers else ""
    else:
        LLVM_RESOURCE_DIR = ""
else:
    LLVM_RESOURCE_DIR = ""

SM = "sm_120"

_CUDA_HOME = os.environ.get("CUDA_HOME", "")
if not _CUDA_HOME:
    for _base in ["/usr/local/cuda", "/opt/cuda"]:
        _matches = sorted(glob.glob(_base + "*"))
        if _matches:
            _CUDA_HOME = _matches[-1]
            break
CUDA_INCLUDE = os.path.join(_CUDA_HOME, "targets", "x86_64-linux", "include") if _CUDA_HOME else ""
INCLUDE_PATHS = [CUDA_INCLUDE] if CUDA_INCLUDE and os.path.isdir(CUDA_INCLUDE) else []

_HAS_LLVM = bool(LLVM_RESOURCE_DIR) and os.path.isdir(LLVM_RESOURCE_DIR) and len(INCLUDE_PATHS) > 0
pytestmark = pytest.mark.skipif(not _HAS_LLVM, reason="LLVM/CUDA headers unavailable")

_compiler_cache = []
_ctx_cache = []


def _compile(source: str, symbol: str):
    compiler = llvm.SuspendedCudaCompiler(
        source=source, opt_level=3, sm=SM,
        resource_dir=LLVM_RESOURCE_DIR, include_paths=INCLUDE_PATHS)
    _compiler_cache.append(compiler)
    ctx = llvm.context()
    _ctx_cache.append(ctx)
    compiler.parse(ctx, "syncguard")
    req = llvm.CudaFuncRequest()
    req.symbol = symbol
    req.param_types = [llvm.ScalarType.Fp32]
    return compiler.compile_bitcode([req])


def test_direct_syncthreads_rejected():
    source = r"""
__device__ void sync_fn(float x) {
    __syncthreads();
}
"""
    ok, bitcode, error, results = _compile(source, "sync_fn")
    assert not ok, "expected compilation to fail for __syncthreads"
    assert "forbidden synchronization primitive" in error
    assert "__syncthreads" in error
    assert "sync_fn" in error


def test_indirect_syncthreads_rejected():
    """__syncthreads reached through a same-TU callee must also be rejected."""
    source = r"""
__device__ void helper_sync() { __syncthreads(); }
__device__ void indirect_fn(float x) { helper_sync(); }
"""
    ok, bitcode, error, results = _compile(source, "indirect_fn")
    assert not ok, "expected compilation to fail for indirect __syncthreads"
    assert "forbidden synchronization primitive" in error
    assert "indirect_fn" in error


def test_threadfence_rejected():
    source = r"""
__device__ void fence_fn(float x) {
    __threadfence();
}
"""
    ok, bitcode, error, results = _compile(source, "fence_fn")
    assert not ok, "expected compilation to fail for __threadfence"
    assert "__threadfence" in error


def test_barrier_sync_rejected():
    source = r"""
__device__ void barrier_fn(float x) {
    __barrier_sync(0);
}
"""
    ok, bitcode, error, results = _compile(source, "barrier_fn")
    assert not ok, "expected compilation to fail for __barrier_sync"
    assert "__barrier_sync" in error


def test_clean_function_accepted():
    source = r"""
__device__ void clean_fn(float x) { }
"""
    ok, bitcode, error, results = _compile(source, "clean_fn")
    assert ok, f"clean function must compile, got error: {error}"
    assert len(bitcode) > 0
