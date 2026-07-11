# Coding Conventions

**Analysis Date:** 2026-07-11

## Naming Patterns

**C++ Files:**
- PascalCase for types, structs, classes: `CUDACompiler`, `TensorTypeHelpers`, `CustomAstConsumer`, `GluonOpBuilder`
- PascalCase for enum names with PascalCase members: `ScalarType` with members `Int32`, `Int64`, `Fp32`, `Fp16`, `Bf16`, `Fp8e4m3`, `Fp8e5m2`
- UPPER_CASE for constants, enum-like constexpr: `kMaxGPR`, `SwitchContextFn`
- snake_case for free functions: `compile_cuda_to_module`, `link_cuda_bitcode`, `extract_extern_call_specs`
- camelCase for member functions: `EvaluateFunctionReturnType`, `BuildLayoutFactory`, `DispatchTypeParsing`
- Header guard: `#pragma once`
- Namespace aliases for brevity: `namespace py = pybind11;`, `namespace tt = triton;`, `namespace ttg = triton::gpu;`

**Python Files:**
- snake_case for functions, methods, variables: `call_extern`, `_pre_compile_extern_calls`, `result_layout`
- PascalCase for classes: `TritonSemantic`, `GluonCallerContext`, `DistributedLayout`, `BlockedLayout`, `CUDABackend`
- Private modules use underscore prefix: `_core.py`, `_semantic.py`, `_layouts.py`, `_runtime.py`
- Private functions use underscore prefix: `_check`, `_compute_result_shape`, `_is_int_list`
- Test functions: `test_` prefix (e.g., `test_elementwise_add`, `test_intra_warp_add_sibling`)

**MLIR Dialects:**
- Dialect names: lowercase dotted (`tt`, `ttg`, `nvws`, `gluon`, `proton`, `protongpu`, `triton_nvidia_gpu`)
- Ops: PascalCase within dialect (`tt.call`, `ttg.extern_call`, `tt.load`, `tt.store`)
- Pass names: kebab-case within dialect prefix (`convert-triton-to-tritongpu`, `tritongpu-pipeline`, `triton-nvidia-mma-lowering`)
- Attributes: PascalCase with prefix (`BlockedEncodingAttr`, `NvidiaMmaEncodingAttr`, `SliceEncodingAttr`)

**File naming:**
- C++: PascalCase for pass/pattern files (`ExternCallOpToLLVM.cpp`), snake_case for general modules (`clang_compiler.cc`, `gluon_ir.cc`)
- Python: snake_case (`test_core.py`, `test_extern_call.py`, `code_generator.py`)
- Headers: same as source file name (`.h` extension for C++: `clang_compiler.h`)

## Code Style

### Python Formatting & Linting

**Primary linter:** `ruff` (v0.9.1), configured in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 120

[tool.ruff.lint]
ignore = ["E501", "E701", "E731", "E741"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
```
- Applied to `python/`, `third_party/proton/`, `third_party/amd/`, `third_party/nvidia/`, `test/` (per `.pre-commit-config.yaml` line 24)
- `--fix --exit-non-zero-on-fix` in pre-commit

**Formatter:** `yapf` (v0.43.0), configured in `pyproject.toml`:
```toml
[tool.yapf]
based_on_style = "pep8"
column_limit = 120
disable_split_list_with_comment = true
each_dict_entry_on_separate_line = false
split_before_named_assigns = false
split_complex_comprehension = true
```
- Run with `-p -i` (parallel, in-place) via pre-commit

**Type checking:** `mypy` (v1.15.0), configured in `pyproject.toml`:
```toml
[tool.mypy]
mypy_path = "$MYPY_CONFIG_FILE_DIR/python"
files = [
    "python/triton/knobs.py",
    "python/triton/runtime/build.py",
    "python/triton/runtime/driver.py",
    "python/triton/_utils.py",
    "python/test/unit/test_knobs.py",
    "python/test/unit/runtime/test_build.py",
    "python/test/unit/runtime/test_compilation_listener.py",
]
exclude = ["/build/"]
follow_imports = "silent"
```
- Only specific files are type-checked; others are excluded via `follow_imports = "silent"`
- Run with `pass_filenames: false` in pre-commit (line 46 of `.pre-commit-config.yaml`)

**Other pre-commit hooks** (`.pre-commit-config.yaml`):
- Standard checks: `check-symlinks`, `destroyed-symlinks`, `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-ast`, `check-added-large-files`, `check-merge-conflict`, `check-executables-have-shebangs`, `check-shebang-scripts-are-executable`, `detect-private-key`, `debug-statements`
- Expanded YAML anchors for GitHub Actions workflow (`integration-tests.yml`)

### C++/CUDA Formatting

**Formatter:** `clang-format` (v19.1.6), configured in `.clang-format`:
```
BasedOnStyle: LLVM
```
- Applied to `.cc`, `.cpp`, `.h`, `.cu` files via pre-commit
- Build flags (from `CMakeLists.txt`):
  - Standard: `-D__STDC_FORMAT_MACROS -fPIC -std=gnu++17`
  - Triton library objects: `-fno-exceptions -fno-rtti` (via `TRITON_DISABLE_EH_RTTI_FLAGS`)
  - `clang_compiler.cc` specifically: `-fno-rtti` (line 460-461 of `CMakeLists.txt`) — required because Clang libs are built without RTTI
  - All code: `-Werror -Wno-covered-switch-default` (Linux, line 294)
  - Hidden visibility by default (unless `TRITON_EXT_ENABLED`)

**Section markers:**
- C++ sections: `// ============================================================` banners with section names (see `clang_compiler.h`, `clang_compiler.cc`)
- Python sections: `# ===---===` style separators (see `semantic.py` lines 35-37)

## Import Organization

**Python import order:**
1. `from __future__ import annotations` (all Python files that use type hints)
2. Standard library imports: `import math`, `from dataclasses import dataclass`
3. Third-party imports: `import numpy as np`, `import torch`, `import pytest`
4. Project imports: `import triton`, `import triton.language as tl`, `from triton._C.libtriton import ir`
5. Local/relative imports: `from . import _core as ttgl`, `from ._layouts import DistributedLayout`

Example from `python/triton/language/core.py`:
```python
from __future__ import annotations
import math
from warnings import warn
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps, cached_property
import typing
from typing import Union, Callable, List, Sequence, TypeVar, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import builtins
from .. import knobs
from ..runtime.jit import JITCallable
import inspect
from .._C.libtriton import ir
```

Example from `python/test/unit/language/test_core.py`:
```python
# ruff: noqa: F821,F841
import contextlib
import itertools
import re
from typing import Optional
import math
import textwrap
import numpy as np
import pytest
import torch
import inspect
from numpy.random import RandomState
import triton
import triton.language as tl
from triton._internal_testing import (...)
```

**Path aliases:** No configured path aliases (no `setup.cfg`, no `[tool.pytest.ini_options]` for pythonpath aliases). Python package structure via `__init__.py` files and relative imports.

## Error Handling

**Python patterns:**
- Standard exception raising with descriptive messages:
  ```python
  raise ValueError(f"program_id axis must be 0, 1, or 2 but got {axis}")
  raise TypeError(f"unexpected signedness {a_sn} and {b_sn}")
  ```
- Custom exception classes: `IncompatibleTypeErrorImpl` (in `semantic.py` line 17-23)
- Triton-specific: `triton.TritonError`, `triton.runtime.errors.PTXASError`, `triton.runtime.errors.InterpreterError`
- Assertions for internal invariants: `assert callable(fn)`, `assert False, f"splitn requires..."`
- Deferred message evaluation via lambda to avoid expensive string formatting in hot paths:
  ```python
  def _check(cond: bool, msg_fn: Callable[[], str], category=ValueError):
      if not cond:
          raise category(msg_fn())
  ```
- `pytest.raises` for testing error conditions (e.g., `test_core.py` line 157: `with pytest.raises(triton.TritonError, match="out of range"):`)

**C++ patterns:**
- `assert()` and `__builtin_unreachable()` for unreachable code paths
- `LogicalResult` (success/failure) for pass and lowering functions (returns `mlir::LogicalResult`)
- `llvm::report_fatal_error()` for unrecoverable conditions
- Scoped diagnostic handling: `ScopedDiagnosticHandler` with lambda capture for error collection (see `unittest/Dialect/TritonGPU/DialectTest.cpp` lines 131-140)
- No exceptions in Triton library code (`-fno-exceptions` compile flag); error handling via return values and diagnostic emission

## Logging

**Framework:** Primarily `print()` and `logging` in Python; `llvm::errs()` and `llvm::outs()` in C++; MLIR diagnostics system via `emitError()`, `emitWarning()`, etc.

**Patterns:**
- Python status messages: `print()` with descriptive context for build steps
- `mlir::emitError(loc, "message")` for MLIR-level error reporting
- `mlir::InFlightDiagnostic` for deferred/filtered diagnostics
- No structured logging framework detected

## Comments

**When to comment:**
- LLVM-style section banners for major divisions in C++ files (see `clang_compiler.h`, `clang_compiler.cc`)
- Docstrings on Python public functions and classes (e.g., `must_use_result`, `builtin`, `patch_kernel`)
- Inline comments explaining non-obvious behavior (e.g., "Same LLVMContext for clone")
- TODO/FIXME markers: found in Python test files (e.g., `test_core.py` line 64: `# TODO: enable multiple cta cluster testing`)

**JSDoc/TSDoc:** Not used (no TypeScript/JavaScript in this codebase).

**MLIR comment conventions:**
- Lit test checks: `// CHECK:`, `// CHECK-LABEL:`, `// CHECK-NEXT:`, `// RUN:` directives
- Debug metadata: `{-# ... #-}` for MLIR reproducer metadata
- ODS descriptions: TableGen doc strings (`let description = ...`, `let summary = ...`)

## Function Design

**Size:** Wide range — compiler pass files can be very large (`test_core.py`: 7095 lines). Individual functions are typically compact (10-50 lines), with helpers factored out. `patch_kernel()` in test utilities, `_test_binary()` / `_test_unary()` generic test helpers.

**Parameters:**
- Python: keyword arguments preferred for optional params, positional for required
- Decorator pattern heavily used: `@triton.jit`, `@builtin`, `@wraps`, `@functools.lru_cache()`, `@gluon.jit`
- `constexpr` pattern for compile-time constants: `SIZE: tl.constexpr`, `BLOCK: tl.constexpr`
- C++: pass large objects by const reference, use `llvm::ArrayRef` for array-like parameters, `llvm::StringRef` for strings

**Return Values:**
- Python: direct return of values, tuples for multiple returns. `None` for void.
- C++: `LogicalResult` for pass/success-failure, `std::optional<>` for nullable, `std::variant<>` for sum types (e.g., `std::variant<std::nullptr_t, TensorParameter, TupleType>`)
- Structured return via `std::tuple<>` for multi-value returns: `std::tuple<std::string, std::string, std::vector<CudaFuncResult>>`

## Module Design

**Exports:** Python uses explicit imports in `__init__.py` files (F401 ignored by ruff to allow import re-exports in `__init__.py`). Public API via `triton.language`, `triton.experimental.gluon.language`.

**Barrel Files:** Python `__init__.py` files serve as barrel files for public module surface. Key examples:
- `python/triton/__init__.py` — top-level API
- `python/triton/language/__init__.py` — language builtins
- `python/triton/experimental/gluon/language/__init__.py` — gluon language API

## pybind11 Binding Patterns

**Module registration** (`python/src/main.cc`):
```cpp
PYBIND11_MODULE(libtriton, m) {
  m.doc() = "Python bindings to the C++ Triton API";
  init_triton_ir(m.def_submodule("ir"));
  init_triton_passes(m.def_submodule("passes"));
  init_triton_llvm(m.def_submodule("llvm"));
  init_gluon_ir(m.def_submodule("gluon_ir"));
  init_linear_layout(m.def_submodule("linear_layout"));
  // ... backend-specific init
}
```

**Class binding pattern** (from `python/src/gluon_ir.cc`, `python/src/llvm.cc`):
```cpp
py::class_<TensorParameter>(m, "TensorParameter")
    .def_readwrite("scalar", &TensorParameter::Type)
    .def_readwrite("shape", &TensorParameter::Shape)
    // ... more properties
```

**Function binding:**
```cpp
m.def("compile_cuda_to_module", &tritonCompileCuda, ...);
m.def("extract_extern_call_specs", &tritonExtractExternCallSpecs, ...);
m.def("link_cuda_bitcode", &linkBitcodeToModule, ...);
```

**Submodules:** Use `m.def_submodule("name")` to organize bindings into categories (ir, passes, llvm, gluon_ir, etc.)

## MLIR Dialect & Pass Patterns

**Dialect definitions:** TableGen ODS files (`.td` extension):
- `include/triton/Dialect/Triton/IR/TritonOps.td` — Triton dialect ops
- `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` — TritonGPU dialect ops
- `include/triton/Dialect/Gluon/IR/` — Gluon dialect
- `include/triton/Dialect/TritonNvidiaGPU/IR/` — NVIDIA-specific GPU dialect

**Pass implementation:** C++ in `lib/Conversion/` and `lib/Dialect/*/Transforms/`:
- Pattern rewriters: subclass `mlir::OpRewritePattern<T>` or `mlir::ConversionPattern`
- Pass creation: `std::make_unique<Pass>()` or `createXxxPass()` factory
- Registration via `PassRegistration` macro

**Pass pipeline:** Python orchestration via ordered dicts of stage extensions (`third_party/nvidia/backend/compiler.py`, class `CUDABackend`), each stage a function `compile_ir(mod, metadata) -> mod`.

**LinearLayout system:** `triton::LinearLayout` is a key abstraction for distributed tensor layout representation; conversions in `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`.

---

*Convention analysis: 2026-07-11*
