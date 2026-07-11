# Technology Stack

**Analysis Date:** 2026-07-11

## Languages

**Primary:**
- C++ 17 - Core compiler infrastructure, MLIR dialect implementations, LLVM passes, clang compiler integration. Set via `CMAKE_CXX_STANDARD 17` in `CMakeLists.txt` and `-std=gnu++17` flag.
- Python 3.x - Public API surface (`python/triton/`), build orchestration (`python/build_helpers.py`, `setup.py`), test suites (`python/test/`), tutorial system (`python/tutorials/`).

**Secondary:**
- CUDA C++ - Device runtime code: `tt_plugin.cu` (extern call device library), `python/triton/experimental/gsan/src/GSanLibrary.cu` (GPU sanitizer runtime). Compiled with clang++ `--cuda-device-only` to LLVM bitcode.
- MLIR TableGen (`.td` / `.tblgen`) - Dialect operation/type/attribute definitions. 34 `.td` files under `include/triton/Dialect/**/IR/` define ODS specifications for `tt`, `ttg`, `ttng`, `ttnvgpu`, `gluon`, and `instrument` dialects.
- PTX Assembly - Intermediate GPU code. NVIDIA backend emits PTX via LLVM NVPTX target, then assembles to cubin via `ptxas`.
- CMake - Build configuration language. Root `CMakeLists.txt` (558 lines) plus per-directory `CMakeLists.txt` files.

## Runtime

**Environment:**
- Python 3.x (interpreter)
- CUDA-capable GPU with NVIDIA driver (`libcuda.so.1`)

**Package Manager:**
- pip / setuptools
- Lockfile: Not present (git-based pinning via `cmake/llvm-info.json` for LLVM, `cmake/nvidia-toolchain-version.json` for CUDA tools)

## Frameworks

**Core:**
- MLIR (part of LLVM monorepo) - Intermediate representation framework. Triton defines its own dialects (`Triton`, `TritonGPU`, `TritonNvidiaGPU`, `Gluon`, `TritonInstrument`) on top of upstream MLIR.
- LLVM 18+ (custom build, pinned at hash `62b7cf9623fc310525f39ed69aaecc318a909731`) - Backend code generation (`NVPTX` target), optimization passes (O3 pipeline), IR translation from MLIR dialects.
- Clang - Used as both **system compiler** (`CC=clang CXX=clang++`) and **JIT compiler library** (clangCodeGen, clangFrontend, clangDriver, clangSema, clangAST for in-process CUDA C++ compilation).
- pybind11 (>=2.13.1) - C++/Python binding layer. All Triton internals exposed to Python via `python/src/main.cc` → `PYBIND11_MODULE(libtriton, m)`.
- setuptools (>=40.8.0) - Python package build and distribution. `setup.py` orchestrates CMake build via custom `CMakeBuild` command.

**Testing:**
- pytest - Primary Python test runner. Configured in `pytest.ini`; run with `pytest-xdist` for parallel execution (`-n 8`).
- Google Test (googletest) - C++ unit tests under `unittest/`. Downloaded during CMake configure via `unittest/googletest.cmake`.
- lit - LLVM Integrated Tester. MLIR-level tests under `test/`. Invoked from build directory: `lit -v test/<path>.mlir`.
- expecttest - Snapshot/expectation testing for Python.

**Build/Dev:**
- CMake (>=3.20, <4.0) - Primary build system generator. Scripted in `setup.py` and `build.sh`.
- Ninja (>=1.11.1) - Build executor (`-G Ninja`). Selected for speed over GNU Make.
- ccache - Optional C/C++ compiler cache (`TRITON_BUILD_WITH_CCACHE:BOOL=ON`).
- lld - LLVM linker. Used via `-DCMAKE_LINKER=ld.lld` and `-fuse-ld=lld` in canonical build.
- pre-commit - Git hooks for linting/formatting. Configured in `.pre-commit-config.yaml`.

## Key Dependencies

**Critical (required for build):**
- pybind11 (>=2.13.1) - Python/C++ bindings. Resolved via `pybind11.get_include()` in setup.py or `PYBIND11_SYSPATH` env var.
- LLVM/MLIR (custom build at `LLVM_SYSPATH`) - Monorepo with MLIR, Clang, LLD, NVPTX backend. Must be self-compiled; Triton's default precompiled LLVM causes symbol mismatches.
- nlohmann/json (v3.11.3, pinned in `cmake/json-version.txt`) - C++ JSON library for extern call spec serialization and metadata. Paths: `JSON_SYSPATH` or auto-downloaded.
- CUDA Toolkit - Provides `ptxas` (PTX assembler), `cuobjdump`, `nvdisasm`, `libcuda.so.1` (driver API), CUDA CRT/Runtime headers (`TRITON_CUDACRT_PATH`, `TRITON_CUDART_PATH`), CUPTI profiler libs.

**Infrastructure:**
- nvidia backend (in-tree at `third_party/nvidia/`) - NVIDIA GPU codegen. Provides `compiler.py` (CUDABackend), `driver.py` (CUDA driver), CUDA-specific MLIR passes (`NVGPUToLLVM`, `TritonNVIDIAGPUToLLVM`), `libdevice.10.bc` (device math).
- proton (in-tree at `third_party/proton/`) - Triton profiler. Requires CUPTI and ROCM headers. Built as separate shared library linked against pybind11.
- f2reduce (in-tree at `third_party/f2reduce/`) - Test case reducer for MLIR.

**Python Runtime:**
- numpy - Array/matrix operations. Used in test suite and triton_kernels library.
- scipy (>=1.7.1) - Scientific computing. Used in tests.

## Configuration

**Environment variables (critical):**
| Variable | Purpose |
|----------|---------|
| `LLVM_SYSPATH` | Path to custom LLVM/MLIR/Clang installation |
| `TRITON_CACHE_PATH` | Triton compilation cache directory (default: `~/.triton/cache`) |
| `TRITON_OFFLINE_BUILD` | Disable internet downloads during build |
| `TRITON_CODEGEN_BACKENDS` | Semicolon-separated list: `nvidia`, `amd` |
| `TRITON_PTXAS_PATH` | Override path to `ptxas` executable |
| `TRITON_PTXAS_BLACKWELL_PATH` | Path to Blackwell-specific `ptxas` |
| `TRITON_CUDACRT_PATH` / `TRITON_CUDART_PATH` | CUDA CRT/Runtime header paths |
| `TRITON_CUPTI_INCLUDE_PATH` / `TRITON_CUPTI_LIB_PATH` | CUPTI profiler paths |
| `PYTHONPATH` | Must include `./python:./third_party/nvidia` for local dev |
| `TRITON_EXT_ENABLED` | Enable LLVM symbol visibility for plugin extensions |
| `TRITON_BUILD_PROTON` | Toggle proton profiler build (default ON) |
| `TRITON_BUILD_UT` | Toggle C++ unit tests (default ON) |
| `MAX_JOBS` | Parallel build job count |

**Build configuration files:**
- `CMakeLists.txt` (558 lines) - Root build configuration
- `pyproject.toml` - Build system requirements, mypy/yapf/ruff config
- `cmake/llvm-info.json` - Pinned LLVM hash and per-platform SHA256 sums
- `cmake/nvidia-toolchain-version.json` - Pinned CUDA tool versions
- `setup.py` (620 lines) - Python packaging, cmake invocation, backend install
- `python/build_helpers.py` (666 lines) - Dependency download, cache management, package resolution
- `build.sh` (33 lines) - Canonical dev build script (cmake + ninja + lld)

**Code style:**
- `.clang-format` - LLVM-based style for C++/CUDA
- `.editorconfig` - 2-space indent for C/CUDA, 4-space for Python, tabs for Makefiles
- `pyproject.toml` - ruff (line-length=120), yapf (pep8, column_limit=120), mypy (selective files)
- `.pre-commit-config.yaml` - Hooks for ruff, yapf, clang-format, mypy

## Platform Requirements

**Development:**
- Linux (x86-64 or ARM64) with CUDA-capable NVIDIA GPU
- clang 18+ as system compiler
- Self-compiled LLVM at `LLVM_SYSPATH` (includes MLIR, Clang, LLD)
- CUDA Toolkit 12+ (ptxas, cuobjdump, nvdisasm)
- Python 3.8+ with numpy, pytest

**Production:**
- Deployment target: CUDA-capable NVIDIA GPU (SM70+)
- Runtime dependency: `libcuda.so.1` (NVIDIA driver), `libdevice.10.bc`
- Wheel distribution via `python -m build` or `pip install`

## Key C++ Source Locations

| Purpose | Path |
|---------|------|
| Python bindings (entry) | `python/src/main.cc` |
| Triton IR bindings | `python/src/ir.cc`, `python/src/ir.h` |
| LLVM IR bindings | `python/src/llvm.cc` |
| Clang compiler integration | `python/src/clang_compiler.cc`, `python/src/clang_compiler.h` |
| MLIR pass bindings | `python/src/passes.cc`, `python/src/passes.h` |
| Gluon IR bindings | `python/src/gluon_ir.cc` |
| Linear layout bindings | `python/src/linear_layout.cc` |
| Interpreter bindings | `python/src/interpreter.cc` |
| Kernel specialization | `python/src/specialize.cc` |
| Triton optimizer tool | `bin/triton-opt.cpp` |
| Triton LSP server | `bin/triton-lsp.cpp` |
| Triton reduce tool | `bin/triton-reduce.cpp` |

---

*Stack analysis: 2026-07-11*
