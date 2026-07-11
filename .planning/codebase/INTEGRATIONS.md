# External Integrations

**Analysis Date:** 2026-07-11

## Compiler Toolchain

### LLVM/MLIR

- **What it's used for:** Core compiler infrastructure. MLIR provides the intermediate representation framework; LLVM provides backend code generation and optimization.
- **Version:** Custom self-compiled, pinned at hash `62b7cf9623fc310525f39ed69aaecc318a909731` (build number 2, defined in `cmake/llvm-info.json`).
- **Path:** `LLVM_SYSPATH` env var → `/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/llvm-data/install` (canonical for this fork).
- **CMake integration:** `find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})`, includes `AddMLIR`, `TableGen`, `AddLLVM`.
- **Key LLVM components used:**
  - `LLVMNVPTXCodeGen` - NVIDIA PTX code generation via NVPTX target
  - `LLVMMIRParser` - Machine IR parsing (permanently linked in `CMakeLists.txt`)
  - `LLVMPasses` - Optimization pass infrastructure
  - `LLVMX86CodeGen` / `LLVMAArch64CodeGen` - Host codegen (for interpreter and host-side tools)
  - `LLD` (linker) - Used at link time (`ld.lld`)
- **Key MLIR components used:** `MLIRNVVMDialect`, `MLIRNVVMToLLVMIRTranslation`, `MLIRGPUToNVVMTransforms`, `MLIRIR`, `MLIRPass`, `MLIRTransforms`, `MLIRLLVMDialect`, `MLIRTargetLLVMIRExport`, `MLIRMathToLLVM`, `MLIRGPUDialect`, `MLIRSCFToControlFlow`, `MLIRIndexToLLVM`, `MLIRUBToLLVM`, `MLIRControlFlowToLLVM`, `MLIRSupport`.
- **Source:** `cmake/llvm-build-info.json`, resolved/configured via `python/build_helpers.py:write_thirdparty_cmake_vars()` → `cmake --build` at `CMakeLists.txt:122-135`.

### Clang Libraries (C++ Frontend)

- **What it's used for:** In-process CUDA C++ compilation for `gl.call()` extern function interop. At JIT time, the compiler parses template CUDA source (`tt_plugin.cu`), performs template argument deduction and overload resolution via Sema, and generates LLVM bitcode via CodeGen — all within the same process.
- **Clang libraries permanently linked** (`CMakeLists.txt:387-395`):
  - `clangCodeGen` - LLVM IR generation from C++ AST (CodeGen::ModuleBuilder)
  - `clangFrontend` - Compiler frontend framework (CompilerInstance, FrontendAction)
  - `clangDriver` - Compiler driver (toolchain, flags)
  - `clangBasic` - Core clang types, diagnostics, source locations
  - `clangSerialization` - AST serialization/deserialization
  - `clangLex` - C++ lexer and preprocessor
  - `clangParse` - C++ parser
  - `clangSema` - Semantic analysis (template deduction, overload resolution, type checking)
  - `clangAST` - Abstract Syntax Tree representation (Decl, Type, TemplateName)
- **Integration details:**
  - `python/src/clang_compiler.cc` - `CUDACompiler` class wraps clang::CompilerInstance with in-memory VFS, `CustomAstConsumer`, and `CustomFEAction` to create `CodeGen::ModuleBuilder`.
  - `python/src/clang_compiler.h` - `TensorTypeHelpers` (template instantiation), `TypeBuilder`/`TypeInspector` (AST ↔ TensorParameter), `FunctionResolver` (Sema overload resolution).
  - Compiled with `-fno-rtti` (`CMakeLists.txt:460-461`) because upstream Clang libraries are built without RTTI.
- **Build context:** `find_package(Clang REQUIRED CONFIG PATHS ${CLANG_DIR})` at `CMakeLists.txt:238`.

### CUDA Toolchain

- **What it's used for:** GPU code assembly (PTX→cubin), binary inspection, and profiling.
- **Tool versions** (pinned in `cmake/nvidia-toolchain-version.json`):

| Tool | Version | Purpose | CMake Var |
|------|---------|---------|-----------|
| ptxas | 12.9.86 | PTX assembler (default for pre-Blackwell) | `TRITON_PTXAS_PATH` |
| ptxas-blackwell | 13.3.33 | PTX assembler for SM100+ (Blackwell) | `TRITON_PTXAS_BLACKWELL_PATH` |
| cuobjdump | 13.1.80 | CUDA binary inspector | `TRITON_CUOBJDUMP_PATH` |
| nvdisasm | 13.1.80 | SASS disassembler | `TRITON_NVDISASM_PATH` |
| cudacrt | 13.1.80 | CUDA C runtime headers | `TRITON_CUDACRT_PATH` |
| cudart | 13.1.80 | CUDA runtime headers | `TRITON_CUDART_PATH` |
| cupti | 12.8.90 | CUDA Profiling Tools Interface | `TRITON_CUPTI_INCLUDE_PATH` / `TRITON_CUPTI_LIB_PATH` |
| cupti-blackwell | 13.3.35 | CUPTI for Blackwell GPUs | `TRITON_CUPTI_LIB_BLACKWELL_PATH` |

- **Runtime:** `libcuda.so.1` (NVIDIA Driver API). Resolved at `third_party/nvidia/backend/driver.py:25-46` via `ldconfig` or `LD_LIBRARY_PATH`.
- **Device math library:** `third_party/nvidia/backend/lib/libdevice.10.bc` - LLVM bitcode library providing GPU math intrinsics (`__nv_sin`, `__nv_exp`, etc.).
- **PTX version mapping:** `third_party/nvidia/backend/compiler.py:51-72` maps CUDA driver version to PTX ISA version. LLVM cap is PTX 9.0 (`llvm_ptx_version = min(90, ptx_version)`).

### pybind11 Bindings

- **What it's used for:** Exposing all C++ Triton internals to Python as a native extension module `triton._C.libtriton`.
- **Version:** >=2.13.1 (build), runtime resolved at import.
- **Module structure:** Single `PYBIND11_MODULE(libtriton, m)` in `python/src/main.cc:51` creates submodule hierarchy:
  - `libtriton.ir` - MLIR IR construction (`python/src/ir.cc`)
  - `libtriton.llvm` - LLVM module operations, optimization, extern call orchestration (`python/src/llvm.cc`)
  - `libtriton.passes` - MLIR pass pipeline construction (`python/src/passes.cc`)
  - `libtriton.interpreter` - Triton interpreter (`python/src/interpreter.cc`)
  - `libtriton.gluon_ir` - Gluon dialect IR builder (`python/src/gluon_ir.cc`)
  - `libtriton.nvidia` - NVIDIA backend dialect loading, passes, utilities (`third_party/nvidia/triton_nvidia.cc`)
  - `libtriton.gsan_testing` - GPU sanitizer test support (`python/triton/experimental/gsan/src/gsan_testing.cc`)
  - `libtriton.linear_layout` - Linear layout utilities (`python/src/linear_layout.cc`)
- **Backend extensibility:** Plugin system via `TRITON_BACKENDS_TUPLE` compile definition and `TRITON_PLUGIN_DIRS` for external backends. Each backend registers via `FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)` macro.

## Codegen Backends

### NVIDIA (CUDA) Backend

- **Location:** `third_party/nvidia/`
- **Python entry:** `third_party/nvidia/backend/compiler.py` (`CUDABackend` class, 743 lines), `third_party/nvidia/backend/driver.py` (403 lines).
- **Compiler pipeline stages** (defined as ordered extensions in `CUDABackend`):
  1. **make_ttir** (lines 265-279): Inlining, canonicalization, CSE, loop unroll
  2. **make_ttgir** (lines 281-359): TT→TTG conversion, coalescing, matmul acceleration, layout conversion, software pipelining, warp specialization
  3. **gluon_to_ttgir** (lines 361-385): Gluon dialect inlining, auto-encoding resolution, TMA lowering
  4. **make_llir** (lines 387-458): TTG→LLVM conversion, shared memory allocation, extern call pre-compilation, NVVM→LLVM conversion
  5. **PTX emission** (lines 459-470): MLIR→LLVM IR translation, `llvm.to_module()`, NVPTX target triple setup
  6. **Optimization** (lines 489-491): O3 LLVM optimization
  7. **cubin assembly** (lines 500+): LLVM bitcode → PTX (NVPTX CodeGen) → cubin (ptxas)
- **C++ MLIR passes:**
  - `lib/Conversion/TritonGPUToLLVM/` (24 `.cpp` files) - Core TTG→LLVM lowering including `ExternCallOpToLLVM.cpp`
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/` (23 `.cpp` files) - NVIDIA-specific TTG→LLVM lowering
  - `third_party/nvidia/lib/NVGPUToLLVM/` - NVGPU dialect→LLVM lowering
  - `third_party/nvidia/lib/Dialect/` - NVWS dialect support
- **C++ Triton-to-TritonGPU conversion:** `lib/Conversion/TritonToTritonGPU/`

### AMD (ROCm) Backend

- **Location:** Referenced in build system (`TRITON_CODEGEN_BACKENDS="nvidia;amd"`), `third_party/amd/` expected as submodule.
- **Status:** Not active in canonical build (`build.sh` sets `TRITON_CODEGEN_BACKENDS="nvidia"` only).

### External Plugin Backends

- **Mechanism:** `TRITON_PLUGIN_DIRS` env var (semicolon-separated paths). Each plugin directory must contain `backend/compiler.py`, `backend/driver.py`, and `backend/name.conf`.
- **CMake integration:** `CMakeLists.txt:342-356` — plugins built as subdirectories, output under `${TRITON_BINARY_DIR}/third_party/${PLUGIN_NAME}`.

## Profiling & Instrumentation

### Proton Profiler

- **Location:** `third_party/proton/`
- **What it's used for:** GPU kernel profiling (timeline, metrics, roofline analysis).
- **Dependencies:** CUPTI (NVIDIA), ROCProfiler-SDK / ROCM headers (AMD), nlohmann/json, pybind11, Python3.
- **Build:** Integrated via `TRITON_BUILD_PROTON=ON` (default). Separate shared library linked against pybind11. Configured at `third_party/proton/CMakeLists.txt`.
- **Python install path:** `triton.profiler` (symlinked to `third_party/proton/proton/`).

### GSan (GPU Sanitizer)

- **Location:** `python/triton/experimental/gsan/src/`
- **What it's used for:** GPU address sanitizer — detects out-of-bounds access, use-after-free on GPU.
- **Runtime:** `GSanLibrary.cu` compiled to LLVM bitcode (`gsan.ll`) via clang++ `--cuda-device-only` targeting SM80. Built at `third_party/nvidia/CMakeLists.txt:6-48`.
- **Instrumentation modes:** `consan` (concurrency sanitizer), `iisan` (indirect index sanitizer), `fpsan` (floating-point sanitizer), `gsan` (address sanitizer). Configured via `CUDAOptions.instrumentation_mode`.

### Line Info / Debug Support

- **Location:** `lib/Target/LLVMIR/LLVMDIScope.cpp`, `LLVMDILocalVariable.cpp`
- **What it's used for:** Emitting LLVM debug info (DI) for backtrace and profiling support. Controlled by `knobs.compilation.disable_line_info` and `knobs.compilation.dump_ir_extract_di_local_variables`.

## Development Tools

### triton-opt

- **Location:** `bin/triton-opt.cpp`
- **What it's used for:** Standalone MLIR optimizer tool. Runs MLIR passes on `.mlir` files. Supports `--run-reproducer` for crash reproduction.
- **Build:** `ninja -C build triton-opt`

### triton-lsp

- **Location:** `bin/triton-lsp.cpp`
- **What it's used for:** Language Server Protocol server for Triton IR. Provides IDE support for `.mlir` files.

### triton-reduce

- **Location:** `bin/triton-reduce.cpp`
- **What it's used for:** MLIR test case reduction (bugpoint-style).

### triton-tensor-layout

- **Location:** `bin/triton-tensor-layout.cpp`
- **What it's used for:** Tensor layout visualization/analysis tool.

## Data Storage

**Databases:**
- None. Compilation cache uses filesystem: `~/.triton/cache/` (configurable via `TRITON_CACHE_PATH`).

**File Storage:**
- Local filesystem only. Wheel output to `TRITON_WHEEL_DIR` (default `/tmp/triton_wheel`).

**Caching:**
- Compilation cache: File-based hash in `python/triton/runtime/cache.py`. Cache key includes source hash, options hash, and compiler version.
- `ccache` for C++ compilation (optional, auto-detected).

## CI/CD & Deployment

**Hosting:**
- GitHub workflows under `.github/workflows/` (integration tests, linting).

**CI Pipeline:**
- `.github/workflows/integration-tests.yml` (expanded from `.in` file via pre-commit YAML anchor expansion).
- pre-commit hooks enforce linting/formatting.

**Development deployment:**
- NOT `pip install -e .` (overwrites venv's standard triton). Instead:
  1. Build: `build.sh` → produces `build/libtriton.so`
  2. Copy: `cp build/libtriton.so python/triton/_C/libtriton.so`
  3. Run: `PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia" python3 ...`

## Webhooks & Callbacks

**Incoming:**
- None. No webhook endpoints.

**Outgoing:**
- Dependency downloads during CMake configure (unless `TRITON_OFFLINE_BUILD=1`): LLVM precompiled binary, nlohmann/json. Fetched via `python/build_helpers.py:90-97` using `urllib.request` with Firefox user agent.

## Environment Configuration

**Critical env vars (must be set for build):**
- `LLVM_SYSPATH` - Path to self-compiled LLVM installation
- `TRITON_CACHE_PATH` - Compilation cache (default `~/.triton/cache`)

**Critical env vars (must be set for runtime/development):**
- `PYTHONPATH` - Must include `$(pwd)/python:$(pwd)/third_party/nvidia`
- `CC=clang`, `CXX=clang++` - Required for build (LLVM built with clang)

**Secrets location:**
- Not applicable (no cloud services, no API keys). All auth is local filesystem / GPU driver.

## Test Infrastructure

**Framework stack:**
- lit (LLVM Integrated Tester): MLIR-level regression tests under `test/`. Configured in `test/lit.cfg.py` and `test/lit.site.cfg.py.in`.
- pytest + pytest-xdist: Python-level tests under `python/test/`. Parallel execution via `-n 8`.
- Google Test: C++ unit tests under `unittest/`. Downloaded during CMake configure.

**Test commands** (from `Makefile`):
```bash
make test-lit          # MLIR lit tests (no GPU needed)
make test-unit         # Python GPU tests (pytest, parallel)
make test-cpp          # C++ unit tests (Google Test)
make test-gluon        # Gluon API tests
make test-regression   # Regression tests
make test-proton       # Profiler tests
make test-interpret    # Interpreter tests
```

**Benchmarking:**
- Python microbenchmarks under `python/test/microbenchmark/`. Entry: `make test-microbenchmark` → `python/test/microbenchmark/launch_overhead.py`.

---

*Integration audit: 2026-07-11*
