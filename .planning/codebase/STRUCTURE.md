# Codebase Structure

**Analysis Date:** 2026-07-11

## Directory Layout

```
triton-cu/
├── .planning/                         # GSD planning artifacts
├── bin/                               # C++ CLI tools (triton-opt, triton-lsp, triton-reduce)
├── cmake/                             # CMake modules (LLVM finder, build helpers)
├── docs/                              # Documentation
├── examples/                          # Example kernels and plugin demo
├── include/triton/                    # C++ public headers (dialect ODS, conversions, analysis)
│   ├── Analysis/                      #   Analysis pass headers
│   ├── Conversion/                    #   Conversion pass headers (tt→ttg, ttg→LLVM)
│   ├── Dialect/                       #   Dialect definitions (tt, ttg, Gluon, TritonNvidiaGPU, TritonInstrument)
│   └── Target/LLVMIR/                 #   LLVM IR target pass declarations
├── lib/                               # C++ library implementations
│   ├── Analysis/                      #   Static analysis passes (alias, allocation, axis info)
│   ├── Conversion/                    #   Dialect conversion passes
│   │   ├── TritonToTritonGPU/         #     tt → ttg lowering
│   │   ├── TritonGPUToLLVM/           #     ttg → LLVM dialect lowering (incl. ExternCallOpToLLVM)
│   │   └── TritonInstrumentToLLVM/    #     Instrumentation → LLVM
│   ├── Dialect/                       #   Dialect implementations
│   │   ├── Triton/                    #     tt dialect (IR + Transforms)
│   │   ├── TritonGPU/                 #     ttg dialect (IR + Transforms + LinearLayoutConversions)
│   │   ├── Gluon/                     #     Gluon dialect (IR + Transforms)
│   │   ├── TritonNvidiaGPU/           #     NVIDIA-specific ops (tensor memory, warp group)
│   │   └── TritonInstrument/          #     Instrumentation dialect
│   ├── Target/LLVMIR/                 #   LLVM IR target utilities (debug info, Phi breaking)
│   └── Tools/                         #   C++ tool support
├── python/                            # Python package (triton)
│   ├── src/                           #   pybind11 C++ bindings
│   │   ├── clang_compiler.cc/.h       #     In-process CUDA C++ compiler (clang CodeGen)
│   │   ├── llvm.cc                    #     Python bindings: ScalarType, TensorParameter, CUDA compilation
│   │   ├── ir.cc/ir.h                 #     MLIR IR bindings (Operation, ModuleOp, set_str_attr)
│   │   ├── gluon_ir.cc               #     GluonOpBuilder: create_extern_call, to/from linear layout
│   │   ├── passes.cc/passes.h         #     MLIR pass registration bindings
│   │   ├── linear_layout.cc           #     LinearLayout Python bindings
│   │   ├── interpreter.cc             #     Triton interpreter bindings
│   │   ├── specialize.cc              #     Kernel specialization bindings
│   │   └── main.cc                    #     Module initialization
│   ├── triton/                        #   Main Python package
│   │   ├── compiler/                  #     Compiler frontend (AST→IR, orchestration)
│   │   ├── language/                  #     Triton DSL builtins (core, math, semantic, standard)
│   │   ├── runtime/                   #     JIT, caching, driver, autotuner
│   │   ├── backends/                  #     Backend abstraction (BaseBackend, GPU target)
│   │   ├── experimental/gluon/        #     Gluon experimental API (gl.call, distributed layouts)
│   │   ├── profiler/                  #     Profiling utilities
│   │   ├── instrumentation/           #     GPU instrumentation lib
│   │   ├── tools/                     #     CLI tools (compile, disasm, link, etc.)
│   │   └── _C/                        #     Build output: libtriton.so copied here
│   ├── test/                          #   Python tests
│   │   ├── unit/                      #     Unit tests (language, runtime, instrumentation)
│   │   ├── gluon/                     #     Gluon-specific tests (incl. test_extern_call.py)
│   │   ├── regression/                #     Regression test suite
│   │   ├── backend/                   #     Backend-specific tests
│   │   └── gsan/                      #     GSAN tests
│   ├── tutorials/                     #   Jupyter/script tutorials
│   └── triton_kernels/                #   Pre-built Triton kernel library
├── third_party/                       # Vendor-specific backends
│   ├── nvidia/                        #   NVIDIA GPU backend
│   │   ├── backend/                   #     Python backend (compiler.py CUDABackend, driver.py)
│   │   ├── lib/                       #     C++ NVIDIA-specific conversion passes
│   │   │   ├── TritonNVIDIAGPUToLLVM/ #       NVIDIA ttg→LLVM lowering (TMA, barriers, tensor memory)
│   │   │   ├── NVGPUToLLVM/           #       NVGPU dialect→LLVM lowering
│   │   │   └── Dialect/               #       NVIDIA-specific dialect (NVWS)
│   │   ├── hopper/                    #     Hopper architecture passes
│   │   ├── language/                  #     Python language extensions
│   │   └── include/                   #     NVIDIA-specific C++ headers
│   ├── amd/                           #   AMD GPU backend (if enabled)
│   ├── proton/                        #   Proton profiler (separate sub-project)
│   │   ├── csrc/                      #     C++ profiler sources
│   │   ├── Dialect/                   #     Proton MLIR dialect
│   │   ├── proton/                    #     Python package
│   │   └── test/                      #     Proton tests
│   └── f2reduce/                      #   F2 reduction library
├── test/                              # MLIR lit tests (C++ or .mlir-based)
│   ├── Triton/                        #   tt dialect tests (.mlir)
│   ├── TritonGPU/                     #   ttg dialect tests (.mlir)
│   ├── TritonNvidiaGPU/               #   NVIDIA-specific dialect tests
│   ├── Gluon/                         #   Gluon dialect tests (.mlir)
│   ├── Conversion/                    #   Conversion pass tests
│   ├── NVWS/                          #   NVWS dialect tests
│   ├── Proton/                        #   Proton dialect tests
│   └── Hopper/                        #   Hopper architecture tests
├── unittest/                          # C++ unit tests (Google Test)
├── scripts/                           # Build helper scripts (build-llvm-project.sh)
├── build.sh                           # Canonical build script (cmake + ninja)
├── CMakeLists.txt                     # Root CMake build config
├── Makefile                           # Dev helper (make test-lit, test-unit, etc.)
├── setup.py                           # setuptools packaging
├── pyproject.toml                     # Python project config (mypy, ruff, yapf)
└── AGENTS.md                          # Developer guide (build, test, architecture overview)
```

## Directory Purposes

**`include/triton/`:**
- Purpose: Public C++ headers for all MLIR dialect definitions, conversion passes, analysis passes, and target utilities
- Contains: TableGen (`.td`) dialect definitions, generated `.h` headers, pass headers
- Key files: `include/triton/Dialect/Triton/IR/TritonOps.td`, `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`, `include/triton/Conversion/TritonGPUToLLVM/`

**`lib/`:**
- Purpose: C++ implementations of MLIR dialects, conversions, analyses, and target support
- Contains: `.cpp` implementation files, CMakeLists.txt for each sub-component
- Key files: `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp`, `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`, `lib/Dialect/Triton/Transforms/`

**`python/src/`:**
- Purpose: pybind11 C++/Python bindings — the bridge between Python compiler and C++ MLIR/LLVM/clang
- Contains: `clang_compiler.cc/.h` (CUDA C++ interop), `llvm.cc` (Python bindings), `gluon_ir.cc` (Gluon op builder), `ir.cc` (MLIR IR bindings)
- Key files: `clang_compiler.cc`, `llvm.cc`, `gluon_ir.cc`

**`python/triton/`:**
- Purpose: Main Triton Python package — user-facing API, compiler frontend, runtime
- Contains: `compiler/`, `language/`, `runtime/`, `backends/`, `experimental/gluon/`, `profiler/`, `tools/`
- Key files: `triton/runtime/jit.py` (JITFunction), `triton/language/core.py` (DSL builtins), `triton/compiler/compiler.py` (compile orchestration), `triton/compiler/code_generator.py` (AST→MLIR)

**`third_party/nvidia/`:**
- Purpose: NVIDIA GPU backend — Python compiler driver, C++ NVIDIA-specific conversion passes, tools
- Contains: `backend/compiler.py` (CUDABackend class), `backend/driver.py`, `lib/TritonNVIDIAGPUToLLVM/` (NVIDIA-specific lowering)
- Key files: `backend/compiler.py`, `lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp`

**`third_party/proton/`:**
- Purpose: Triton Proton profiler — a separate sub-project for GPU kernel profiling
- Contains: `csrc/` (C++ profiler), `Dialect/` (Proton MLIR dialect), `proton/` (Python package), `test/`
- Key files: `csrc/lib/Profiler/`, `Dialect/lib/Dialect/Proton/`

**`bin/`:**
- Purpose: C++ CLI tools for MLIR pass testing, optimization, and development
- Contains: `triton-opt.cpp` (MLIR optimizer), `triton-lsp.cpp` (language server), `triton-reduce.cpp` (IR reducer), `triton-tensor-layout.cpp`
- Key files: `triton-opt.cpp`, `triton-lsp.cpp`

**`test/`:**
- Purpose: MLIR lit tests — `.mlir` files testing dialect operations and transformation passes
- Contains: `Triton/` (tt dialect tests), `TritonGPU/` (ttg dialect tests), `Gluon/`, `Conversion/`
- Key files: `test/Triton/ops.mlir`, `test/TritonGPU/loop-pipeline.mlir`, `test/Gluon/auto_encoding.mlir`

**`python/test/`:**
- Purpose: Python GPU tests using pytest
- Contains: `unit/language/` (DSL builtin tests), `gluon/test_extern_call.py` (E2E gl.call test), `regression/`
- Key files: `python/test/gluon/test_extern_call.py`, `python/test/gluon/tt_plugin.cu`

**`examples/`:**
- Purpose: Example Triton kernels and plugin demonstrations
- Contains: `plugins/` (dialect plugin example), various kernel examples

## Key File Locations

**Entry Points:**
- `python/triton/runtime/jit.py:935`: `triton.jit()` — user-facing kernel decorator
- `python/triton/runtime/jit.py:628`: `JITFunction` class — wraps and launches compiled kernels
- `python/triton/compiler/compiler.py:52`: `ASTSource` class — compiles Python AST to MLIR
- `python/triton/experimental/gluon/language/_core.py:774`: `gl.call()` — extern CUDA C++ interop
- `bin/triton-opt.cpp`: Standalone MLIR optimizer CLI
- `python/src/main.cc`: Python module initialization

**Configuration:**
- `CMakeLists.txt`: Root build configuration (LLVM path, CUDA paths, Clang lib linkage)
- `build.sh`: Canonical build script (CC=clang, LLVM_SYSPATH, cmake + ninja)
- `setup.py`: Python package build and install
- `pyproject.toml`: Python tooling config (ruff, yapf, mypy)
- `Makefile`: Dev helper targets (test-lit, test-unit, test-gluon)
- `pytest.ini`: Python test configuration

**Core Logic:**
- `python/triton/language/core.py`: Triton DSL builtins (load, store, arange, reduce, etc.)
- `python/triton/compiler/code_generator.py:286`: `CodeGenerator` class — Python AST→MLIR tt dialect
- `python/triton/compiler/compiler.py`: `compile()` — top-level compilation orchestration
- `third_party/nvidia/backend/compiler.py:185`: `CUDABackend` — NVIDIA compilation pipeline (5 stages)
- `python/src/clang_compiler.cc`: In-process clang CodeGen for CUDA C++ templates

**MLIR Dialect Definitions:**
- `include/triton/Dialect/Triton/IR/TritonOps.td`: tt dialect operations (ODS)
- `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`: ttg dialect operations (ODS)
- `include/triton/Dialect/Gluon/IR/GluonOps.td`: Gluon dialect (extern_call, etc.)
- `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`: Layout encoding attribute definitions

**MLIR Pass Implementations:**
- `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp`: tt→ttg conversion
- `lib/Conversion/TritonGPUToLLVM/`: ttg→LLVM base lowering (ExternCallOpToLLVM, ElementwiseOpToLLVM, etc.)
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp`: NVIDIA-specific ttg→LLVM

**Testing:**
- `test/`: MLIR lit tests (`.mlir` files, run via `make test-lit`)
- `python/test/`: Python GPU tests (run via `make test-unit`, `make test-gluon`)
- `python/test/gluon/test_extern_call.py`: E2E test for `gl.call()` with `tt_plugin.cu`
- `unittest/`: C++ unit tests (Google Test framework)

**Gluon / Extern CUDA Interop:**
- `python/triton/experimental/gluon/language/_core.py:774`: `gl.call()` user-facing API
- `python/triton/experimental/gluon/language/_semantic.py:237`: `call_extern()` — builds MLIR IR types
- `python/triton/experimental/gluon/language/_layouts.py`: DistributedLayout Python types
- `python/triton/experimental/gluon/_runtime.py`: Cache key generation (includes .cu path)
- `python/src/gluon_ir.cc:191-280`: `layoutToGluon()` — MLIR encoding→Python DistributedLayout
- `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp`: ttg.extern_call→LLVM lowering
- `python/src/clang_compiler.cc`: CUDA C++ to LLVM bitcode compilation
- `python/src/llvm.cc`: Python bindings for extern compilation/linking
- `python/test/gluon/tt_plugin.cu`: Reference CUDA device library (Shape, TensorLayout, Tensor templates)

## Naming Conventions

**Files:**
- C++ headers: `PascalCase.h` (e.g., `Ops.h`, `Dialect.h`, `Traits.h`, `Utility.h`)
- C++ sources: `PascalCase.cpp` (e.g., `ExternCallOpToLLVM.cpp`, `LinearLayoutConversions.cpp`)
- TableGen: `PascalCase.td` (e.g., `TritonOps.td`, `TritonGPUAttrDefs.td`, `GluonDialect.td`)
- Python: `snake_case.py` (e.g., `code_generator.py`, `clang_compiler.cc`)
- MLIR test files: `kebab-case.mlir` (e.g., `loop-pipeline.mlir`, `auto_encoding.mlir`)
- Test files: `test_snake_case.py` (e.g., `test_extern_call.py`, `test_core.py`)

**Directories:**
- MLIR dialect directories: `PascalCase/` (`Triton/`, `TritonGPU/`, `Gluon/`, `TritonNvidiaGPU/`)
- Conversion directories: `PascalCase/` (`TritonToTritonGPU/`, `TritonGPUToLLVM/`)
- Python packages: `snake_case/` (`triton/`, `experimental/`, `gluon/`)
- Private Python modules: `_underscore_prefix.py` (`_core.py`, `_semantic.py`, `_layouts.py`)
- Third-party directories: `lowercase/` (`nvidia/`, `amd/`, `proton/`, `f2reduce/`)

**Functions/Methods:**
- Python: `snake_case` (e.g., `ast_to_ttir`, `make_llir`, `call_extern`, `_pre_compile_extern_calls`)
- C++: `PascalCase` or `camelCase` following LLVM conventions (e.g., `convertTritonGPUToLLVM`, `CloneFunctionInto`)
- MLIR pass names: `kebab-case` (e.g., `convert-triton-to-tritongpu`, `convert-triton-gpu-to-llvm`)

**Classes:**
- Python: `PascalCase` (e.g., `JITFunction`, `CUDABackend`, `CodeGenerator`, `ASTSource`)
- C++: `PascalCase` (e.g., `CUDACompiler`, `TensorTypeHelpers`, `CustomAstConsumer`, `GluonOpBuilder`)

## Where to Add New Code

**New Triton Language Builtin:**
- Primary code: `python/triton/language/core.py` (add `@builtin` decorated function)
- Semantic handling: `python/triton/language/semantic.py` (add semantic analysis)
- C++ lowering: `lib/Conversion/TritonGPUToLLVM/` (add `<OpName>OpToLLVM.cpp`)
- MLIR lit tests: `test/Triton/` (add `.mlir` test)
- Python tests: `python/test/unit/language/` (add `test_case.py`)

**New MLIR Dialect Pass:**
- Pass declaration: `include/triton/Dialect/<Dialect>/Transforms/` (`.td` file + header)
- Pass implementation: `lib/Dialect/<Dialect>/Transforms/` (`.cpp` file)
- Python bindings: `python/src/passes.cc` (register in `init_triton_passes_ttir`)
- Tests: `test/<Dialect>/` (`.mlir` lit tests)

**New Conversion Pass:**
- Header: `include/triton/Conversion/<PassName>/`
- Implementation: `lib/Conversion/<PassName>/`
- Lit tests: `test/Conversion/<PassName>/` (`.mlir` files)

**New Python Tool:**
- Implementation: `python/triton/tools/<tool_name>.py`
- If it has submodules: `python/triton/tools/<tool_name>/__init__.py`

**New Gluon Feature:**
- Language API: `python/triton/experimental/gluon/language/_core.py`
- Semantic handling: `python/triton/experimental/gluon/language/_semantic.py`
- Layout types: `python/triton/experimental/gluon/language/_layouts.py`
- C++ op builder: `python/src/gluon_ir.cc`
- Dialect definition: `include/triton/Dialect/Gluon/IR/GluonOps.td`
- Tests: `python/test/gluon/` (pytest) and `test/Gluon/` (lit)

**New Extern CUDA Function (`tt_plugin.cu`):**
- Primary code: Add template function to `python/test/gluon/tt_plugin.cu` (or a new `.cu` file)
- E2E test: `python/test/gluon/test_extern_call.py` (pytest with Triton kernel using `gl.call()`)

## Special Directories

**`python/triton/_C/`:**
- Purpose: Build output directory for `libtriton.so` — the compiled C++/MLIR shared library
- Generated: Yes (by `ninja -C build triton`)
- Committed: No (build artifact, in `.gitignore`)

**`.planning/`:**
- Purpose: GSD workflow artifacts — planning documents, codebase maps, milestones
- Generated: Yes (by GSD commands)
- Committed: Yes

**`build/`:**
- Purpose: CMake build output directory
- Generated: Yes (by `build.sh` or `cmake`)
- Committed: No (build artifact)

**`docs/`:**
- Purpose: Project documentation (not code documentation)
- Generated: No
- Committed: Yes

---

*Structure analysis: 2026-07-11*
