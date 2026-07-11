# Testing Patterns

**Analysis Date:** 2026-07-11

## Test Framework

### Python (GPU/CPU) Tests

**Runner:** `pytest` (via `python -m pytest`), configured in `pytest.ini`:
```ini
[pytest]
addopts = --tb=short
markers = interpreter: indicate whether interpreter supports the test
python_files = test_*.py *_test.py tutorials/*.py examples/*.py
```

**Plugins:** `pytest-instafail` (fails fast on first error unless pytest-sugar is active, see `conftest.py` lines 5-8), `pytest-xdist` (parallel execution via `-n 8`).

**Assertion libraries:**
- `torch.testing.assert_close(out, expected)` — primary for GPU tensor comparisons (e.g., `test_extern_call.py` lines 68, 78, 87, 99-100)
- `np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)` — for CPU numpy comparisons (e.g., `test_core.py` line 186)
- `torch.testing.assert_close` from test helpers (`print_helper.py` line 5)
- Standard `pytest.raises` for error testing (e.g., `test_core.py` line 157)

**Test requirements** (`python/test-requirements.txt`):
```
autopep8
isort
numpy
pytest
pytest-instafail
pytest-xdist
scipy>=1.7.1
llnl-hatchet
pandas<3.0
expecttest
msgpack
```

### MLIR (lit) Tests

**Runner:** LLVM `lit` test runner, configured via `test/lit.cfg.py` and `test/lit.site.cfg.py.in`.
- Test source root: `test/` directory
- Test exec root: `$BUILD_DIR/test`
- File extensions: `.mlir`, `.ll`
- Tools: `triton-opt`, `triton-llvm-opt`, `mlir-translate`, `llc`
- Verification: `FileCheck` with `--enable-var-scope`

### C++ Unit Tests

**Runner:** `gtest` / `gmock` (Google Test, Google Mock), integrated via `AddTritonUnitTest` CMake module.
- Test files in `unittest/` directory:
  - `unittest/Dialect/TritonGPU/DialectTest.cpp`, `LinearLayoutConversionsTest.cpp`, `SwizzleTest.cpp`, etc.
  - `unittest/Dialect/TritonNvidiaGPU/NvmmaSmemAttrsTest.cpp`
  - `unittest/Analysis/UtilityTest.cpp`
  - `unittest/Tools/LinearLayoutTest.cpp`, `LayoutUtilsTest.cpp`

## Run Commands

```bash
make test-lit         # All MLIR lit tests (no GPU needed, from build dir)
make test-cpp         # All C++ unit tests (gtest, no GPU needed)
make test-unit        # Python GPU unit tests (pytest -n 8, parallel)
make test-regression  # Python regression tests
make test-gluon       # Gluon/gl.call tests (pytest -n 8)
make test-proton      # Proton profiler tests (pytest -n 8, with serial hw trace)
make test-python      # All Python tests: unit + plugins + regression + interpret + proton
make test-nogpu       # No-GPU tests: lit + cpp + frontend Python tests
make test-interpret   # Interpreter-mode tests (TRITON_INTERPRET=1)
make test             # All tests: test-lit + test-cpp + test-python
```

**Run single lit test:**
```bash
cd $BUILD_DIR && ninja triton-opt && lit -v test/Triton/ops.mlir
```

**Run single Python test:**
```bash
pytest -s --tb=short python/test/unit/language/test_core.py::test_scalar_overflow
```

**Run E2E extern call test:**
```bash
pytest python/test/gluon/test_extern_call.py
```

**Run lit test for specific dialect:**
```bash
cd $BUILD_DIR && ninja triton-opt && lit -v test/TritonGPU/
```

**View coverage:** Not configured — no `coverage.py` or `--cov` settings detected.

## Test File Organization

### MLIR Lit Tests (`test/` directory)

**Location:** Co-located by dialect/component within `test/`:
```
test/
├── Triton/               # tt dialect tests (ops, canonicalize, loop-unroll, etc.)
├── TritonGPU/            # ttg dialect tests (pipeline, coalesce, matmul, etc.)
│   └── amd/              # AMD-specific ttg tests
├── TritonNvidiaGPU/      # NVIDIA-specific GPU tests (tma, mma, tmem, etc.)
├── Gluon/                # Gluon dialect tests (auto_encoding, inlining, etc.)
├── Proton/               # Proton profiler dialect tests
│   ├── amd/              # AMD proton tests
│   └── nvidia/           # NVIDIA proton tests
├── NVWS/                 # NVIDIA Warp Specialization tests
├── LLVMIR/               # LLVM IR lowering tests
├── Analysis/             # Analysis pass tests
│   └── amd/              # AMD-specific analysis tests
└── Plugins/              # Plugin system tests
```

**Naming:** `*.mlir` files with descriptive names (e.g., `ops.mlir`, `canonicalize.mlir`, `loop-pipeline.mlir`). `*.mlir.in` for input templates that get `FileCheck` annotations added via `make golden-samples`.

**Structure of a lit test:**
```mlir
// RUN: triton-opt %s -pass-name | FileCheck %s
// RUN: triton-opt %s -split-input-file -pass-name -other-pass | FileCheck %s --check-prefix=ALT

// CHECK-LABEL: @function_name
tt.func @function_name(...) {
  // CHECK: op_name
  %0 = tt.op_name ...
  tt.return
}
```

### Python Tests (`python/test/` directory)

**Location:** By concern/feature:
```
python/test/
├── conftest.py                         # Global fixtures (device, fresh_triton_cache, etc.)
├── unit/
│   ├── language/                       # Core language tests
│   │   ├── test_core.py                # 7095 lines, exhaustive op coverage
│   │   ├── test_standard.py            # Standard math functions
│   │   ├── test_random.py              # Random number generation
│   │   ├── test_matmul.py              # Matrix multiplication
│   │   ├── test_block_pointer.py       # Block pointer ops
│   │   ├── test_tuple.py               # Tuple support
│   │   ├── test_line_info.py           # Line info / debug
│   │   ├── test_subprocess.py          # Subprocess launching
│   │   ├── test_compile_errors.py      # Error reporting tests
│   │   ├── print_helper.py             # Helper for print tests
│   │   └── ...                         # 20+ test files
│   ├── runtime/                        # Runtime tests (cache, build, specialize)
│   ├── instrumentation/                # GPU instrumentation tests
│   ├── plugins/                        # Plugin system tests
│   └── tools/                          # Tool tests (slice_kernel, etc.)
├── gluon/                              # Gluon/gl.call tests
│   ├── test_extern_call.py             # E2E extern call test
│   ├── test_frontend.py                # Frontend-only tests (no GPU)
│   ├── test_layout_format_view.py      # Layout format tests
│   ├── test_consan.py                  # Concurrency sanitizer tests
│   └── tt_plugin.cu                    # CUDA C++ device code for extern_call tests
├── regression/                         # Regression test suite
├── microbenchmark/                     # Microbenchmarks (e.g., launch_overhead.py)
└── gsan/                               # Global sanitizer tests
```

**Naming:** Files: `test_*.py` or `*_test.py`. Functions: `test_` prefix (e.g., `test_elementwise_add`, `test_scalar_overflow`).

### C++ Unit Tests (`unittest/` directory)

```
unittest/
├── Dialect/
│   ├── TritonGPU/             # TritonGPU dialect tests
│   └── TritonNvidiaGPU/       # NVIDIA GPU dialect tests
├── Analysis/                  # Analysis pass tests
└── Tools/                     # Tool tests (LinearLayout)
```

**Naming:** `*Test.cpp` files. Test fixtures: `InferLayoutTest`, `SwizzleTest`, etc. GTest macros: `TEST_F`, `EXPECT_EQ`, `EXPECT_TRUE`.

## Test Structure

### Typical Python Test Pattern

```python
import torch
import pytest
import triton
from triton._internal_testing import is_cuda

pytestmark = pytest.mark.skipif(not is_cuda(), reason="CUDA-only test")

@triton.jit
def my_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + off)
    out = x * 2
    tl.store(out_ptr + off, out)

@pytest.mark.parametrize("BLOCK", [128, 256, 512])
def test_my_kernel(BLOCK):
    torch.set_default_device('cuda')
    x = torch.randn((BLOCK,), dtype=torch.float32)
    out = torch.empty_like(x)
    my_kernel[(1,)](x, out, BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x * 2)
```

### Test Fixtures (`python/test/conftest.py`)

```python
@pytest.fixture
def device(request):
    return request.config.getoption("--device")

@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs
        with knobs.cache.scope(), knobs.runtime.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir

@pytest.fixture
def fresh_knobs():
    # Resets all knobs except build, nvidia, amd
    ...

@pytest.fixture
def with_allocator():
    # Sets up the default allocator for the test
    ...
```

### Generic Test Helpers (from `test_core.py`)

```python
def _test_unary(dtype_x, expr, numpy_expr=None, device='cuda', num_ctas=1):
    """Generic unary op test pattern. Patches GENERATE_TEST_HERE in kernel."""
    check_type_supported(dtype_x, device)

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # ... numpy reference + triton run + comparison
```

### Glue Pattern: `patch_kernel()` (from `test_core.py` lines 96-110)

Template kernels are defined with placeholder strings that get replaced at test time:
```python
def patch_kernel(template, to_replace):
    if is_interpreter():
        # Interpret mode: exec the patched source
        ...
    else:
        kernel = triton.JITFunction(template.fn)
        src = kernel.src
        for key, value in to_replace.items():
            src = src.replace(key, value)
        kernel._unsafe_update_src(src)
        return kernel
```

### C++ Test Pattern (gtest/gmock)

```cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace mlir::triton::gpu {
namespace {

class InferLayoutTest : public ::testing::Test {
public:
  InferLayoutTest() : inferLayout(...) {}
protected:
  static MLIRContext ctx;
  DialectInferLayoutInterface *inferLayout;
};

/*static*/ MLIRContext InferLayoutTest::ctx;

void testReshape(RankedTensorType srcTy, RankedTensorType dstTy, ...) {
  ScopedDiagnosticHandler scopedHandler(
      ctx, [&](Diagnostic &diag) { diags.push_back("  - " + diag.str()); });
  result = inferLayout->inferReshapeOpEncoding(...);
  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(inferredEnc, expectedEnc);
}

} // namespace
} // namespace mlir::triton::gpu
```

## Mocking

**Framework:** No dedicated mocking framework detected. Hand-rolled test doubles used:
- `python/test/unit/language/test_line_info.py`: `@pytest.fixture(autouse=True)` for automatic fixture injection
- `python/test/unit/runtime/test_specialize.py`: Manual mock functions (`mock_tensor_from_tensor`, `mock_jit_callable`)
- `python/test/unit/runtime/test_cache.py`: `test_async_compile_mock` — hand-rolled mocking
- `python/test/unit/tools/test_slice_kernel.py`: `mock_kernel` decorator/function for replacing kernel dispatch
- `python/test/unit/test_knobs.py`: Uses `knobs.nvidia.mock_ptx_version` for testing PTX version-dependent behavior

**What to mock:** External dependencies (PTX version, CUDA driver calls), kernel compilation paths.
**What NOT to mock:** Core triton ops, numeric correctness — these run against actual GPU/codegen.

## Fixtures and Factories

**Test data generators** (from `python/triton/_internal_testing.py`):
```python
from triton._internal_testing import (
    numpy_random,         # deterministically seeded random data
    to_triton,            # numpy-to-triton tensor conversion
    to_numpy,             # triton-to-numpy tensor conversion
    dtypes,               # dtype lists
    int_dtypes,           # integer dtypes
    float_dtypes,         # float dtypes
    is_cuda,              # platform checks
    is_hopper,            # GPU capability checks
    # ...
)
```

**Location:** Shared test helpers in `python/triton/_internal_testing.py`, accessed as `from triton._internal_testing import ...`.

**CUDA C++ test plugins:** Test code in CUDA C++ lives alongside test files (e.g., `python/test/gluon/tt_plugin.cu` contains device functions: `elementwise_add`, `intra_warp_add_sibling`, `reduce`, `split_add`).

## Coverage

**Requirements:** None enforced — no coverage tool or threshold configured. No `coverage.py`, `--cov` flag, or Codecov integration detected.

## Test Types

### Unit Tests (Python)

- **Scope:** Individual triton ops, dtype combinations, edge cases
- **Approach:** `@triton.jit` kernels, torch tensors as input/output, numpy for reference computation, `torch.testing.assert_close` for validation
- **Parallel execution:** `pytest -n 8` via `pytest-xdist`
- **Key files:** `python/test/unit/language/test_core.py` (7095 lines), `test_standard.py`, `test_random.py`, `test_block_pointer.py`

### Integration Tests (Python)

- **Scope:** End-to-end compiler pipeline, runtime caching, plugin loading
- **Approach:** Full compilation paths, cache key verification, multi-file/project tests
- **Key files:** `python/test/gluon/test_extern_call.py` (gl.call with in-process CUDA), `python/test/unit/runtime/test_cache.py`, `python/test/unit/plugins/test_plugin.py`

### E2E Tests

- **Framework:** pytest (no separate E2E framework)
- **Key test:** `python/test/gluon/test_extern_call.py` — verifies the full `gl.call()` pipeline: Triton kernel → clang compilation of CUDA C++ → LLVM bitcode linking → PTX generation → cubin execution. Tests multiple return patterns: single return, tuple return, sliced return.
- **CUDA device code:** `python/test/gluon/tt_plugin.cu` — templated device functions compiled at JIT time.

### Lit Tests (MLIR)

- **Scope:** Dialect operations, pass correctness, MLIR transformations
- **Approach:** `// RUN:` directives invoke `triton-opt` with specific passes, `FileCheck` verifies output IR
- **`--split-input-file`:** Common pattern to test multiple scenarios in one file (each section separated by `// -----`)
- **Invalid input tests:** `--verify-diagnostics` flag checks that invalid MLIR produces expected errors
- **Crash tests:** `not --crash triton-opt ... 2>&1 | FileCheck %s` to verify expected crash behavior

### Interpreter Tests

- **Scope:** Triton interpreter mode validation
- **Approach:** Run subset of unit tests with `TRITON_INTERPRET=1`, targeting CPU
- **Run command:** `make test-interpret`
- **Marker:** `@pytest.mark.interpreter` marks tests as interpreter-compatible

## Common Patterns

### Async/GPU Testing

```python
@pytest.mark.parametrize("BLOCK", [512])
def test_elementwise_add(BLOCK):
    torch.set_default_device('cuda')
    x = torch.randn((BLOCK,), dtype=torch.float32)
    out = torch.empty_like(x)
    elementwise_add_kernel[(1,)](x, y, out, num_warps=1)
    torch.cuda.synchronize()  # Wait for GPU
    torch.testing.assert_close(out, x + y)
```

### Error Testing

```python
def test_scalar_overflow(device):
    @triton.jit
    def kernel():
        huge_int: tl.constexpr = 0xFFFFFFFFFFFFFF
        x = tl.full((), 32, dtype=tl.int32)
        y = x + huge_int

    with pytest.raises(triton.TritonError, match="out of range"):
        kernel[(1, )]()
```

### Platform/Capability Gating

```python
pytestmark = pytest.mark.skipif(not is_cuda(), reason="CUDA-only test")

def check_cuda_or_hip(device):
    if device not in ['cuda']:
        pytest.skip("Only for cuda or HIP")

def check_type_supported(dtype, device):
    if device in ['cuda']:
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8 and (dtype is tl.bfloat16 or ...):
            pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")
```

### MLIR Verifier Testing

```mlir
// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func @bad_op(...) {
  // expected-error @+1 {{message}}
  %0 = tt.invalid_op ...
}
```

### Reproducing Compiler Crashes

Compiler crashes sometimes print an MLIR reproducer block. To reproduce:
```
Save the full MLIR + {-# ... #-} metadata to /tmp/file.mlir
Run: triton-opt /tmp/file.mlir --run-reproducer
```

---

*Testing analysis: 2026-07-11*
