<!-- refreshed: 2026-07-11 -->
# Architecture

**Analysis Date:** 2026-07-11

## System Overview

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          Python User API Layer                                  │
├────────────────────┬─────────────────────┬─────────────────────────────────────┤
│  triton.language   │  triton.runtime     │  triton.experimental.gluon          │
│  `python/triton/   │  `python/triton/    │  `python/triton/experimental/gluon/ │
│   language/`       │   runtime/jit.py`   │   language/`                        │
└────────┬───────────┴──────────┬──────────┴───────────┬─────────────────────────┘
         │                      │                       │
         ▼                      ▼                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                         Compiler Orchestration Layer                            │
├────────────────────────────────────────────────────────────────────────────────┤
│  ASTSource (compiler.py) → code_generator.py → ast_to_ttir()                   │
│  CUDABackend (third_party/nvidia/backend/compiler.py)                          │
│  Gluon compiler (_compiler.py)                                                 │
└────────┬───────────────────────────────────────────────────────┬───────────────┘
         │                                                       │
         ▼                                                       ▼
┌────────────────────────────────┐   ┌───────────────────────────────────────────┐
│      MLIR C++ Compiler         │   │      Extern CUDA C++ Interop              │
│   `lib/` + `include/triton/`   │   │  `python/src/clang_compiler.cc`           │
├────────────────────────────────┤   │  `lib/Conversion/TritonGPUToLLVM/`        │
│ Dialects:                      │   │   ExternCallOpToLLVM.cpp                  │
│  • tt (Triton)                 │   │  `python/src/gluon_ir.cc`                 │
│  • ttg (TritonGPU)             │   │  `python/src/llvm.cc`                     │
│  • Gluon                       │   │                                            │
│  • TritonNvidiaGPU             │   │  in-process clang CodeGen → bitcode      │
│ Conversions:                   │   │  CloneFunctionInto + O3 inlining          │
│  • tt → ttg                    │   │                                            │
│  • ttg → LLVM                  │   │                                            │
└───────────────┬────────────────┘   └──────┬────────────────────────────────────┘
                │                            │
                ▼                            ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                          LLVM / PTX / CUDA Binary Output                        │
├────────────────────────────────────────────────────────────────────────────────┤
│  llir → LLVM IR → ptx (PTX assembly) → cubin (CUDA binary `ptxas`)            │
│  `lib/Target/LLVMIR/`  `third_party/nvidia/backend/compiler.py:make_ptx/cubin` │
└────────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA GPU Driver (`third_party/nvidia/backend/driver.c`)    │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `JITFunction` | User-facing decorator for Triton kernels; manages caching, compilation, launch | `python/triton/runtime/jit.py:628` |
| `ASTSource` | Wraps a Python function, provides hash key and IR generation entry point | `python/triton/compiler/compiler.py:52` |
| `CodeGenerator` | Python AST walker that converts Triton-kernel Python AST into MLIR `tt` dialect | `python/triton/compiler/code_generator.py:286` |
| `CUDABackend` | NVIDIA GPU compiler pipeline: orchestrates all lowering stages (ttir→cubin), extern call compilation | `third_party/nvidia/backend/compiler.py:185` |
| `GluonCompiler` | Compiler orchestrator for Gluon (experimental) dialect | `python/triton/experimental/gluon/_compiler.py` |
| `clang_compiler` | In-process CUDA C++ template instantiation and code generation via clang | `python/src/clang_compiler.cc` |
| `ExternCallOpToLLVM` | Lowers `ttg.extern_call` MLIR ops to LLVM dialect (by-pointer arg convention) | `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` |
| `GluonOpBuilder` | C++ builder for Gluon MLIR ops (extern_call, layout conversions) | `python/src/gluon_ir.cc` |
| `llvm.cc bindings` | Python↔C++ FFI for ScalarType, TensorParameter, CUDA compilation, bitcode linking | `python/src/llvm.cc` |
| `triton-opt` | Standalone MLIR optimizer CLI for testing passes | `bin/triton-opt.cpp` |
| `triton-lsp` | Language server for Triton MLIR | `bin/triton-lsp.cpp` |

## Pattern Overview

**Overall:** MLIR-based ahead-of-time (JIT) compiler with dialect lowering pipeline

**Key Characteristics:**
- Python-side `@triton.jit` decorator captures function AST, lowering triggers on first invocation
- Five-stage compilation pipeline: ttir → ttgir → llir → ptx → cubin
- Each stage is implemented as an extension in an ordered dict of `compile_ir(mod, metadata) -> mod` functions
- MLIR is the intermediate representation throughout most of the pipeline; LLVM IR only at the final backend stages
- Extern CUDA C++ interop (`gl.call()`) compiles device code via in-process clang CodeGen and links bitcode into the final LLVM module
- Backend-specific code lives under `third_party/nvidia/` and `third_party/amd/`
- Dialect definitions use MLIR TableGen (`.td` files) for ODS; implementations are in `.cpp` files

## Layers

**Python User API Layer:**
- Purpose: Provides the `@triton.jit` decorator, language builtins (`triton.language`), and the `gl.call()` Gluon API
- Location: `python/triton/language/`, `python/triton/experimental/gluon/language/`
- Contains: Python DSL for GPU kernels, type system, JIT function wrapper
- Depends on: Python runtime layer, C++ bindings (`python/src/`)
- Used by: End-user Triton kernels

**Python Runtime Layer:**
- Purpose: JIT compilation cache management, autotuner, driver abstraction, kernel launch
- Location: `python/triton/runtime/`
- Contains: `jit.py` (JITFunction class, DependenciesFinder), `autotuner.py`, `cache.py`, `driver.py`, `build.py`
- Depends on: Compiler orchestration, C++ bindings
- Used by: User API layer

**Compiler Orchestration Layer:**
- Purpose: Drives the multi-stage compilation pipeline; invokes MLIR passes and external tools (ptxas)
- Location: `python/triton/compiler/compiler.py`, `third_party/nvidia/backend/compiler.py`
- Contains: `ASTSource`, `CUDABackend`, `GluonCompiler`, `compile()`
- Depends on: MLIR C++ compiler via `triton._C.libtriton`
- Used by: Runtime layer (`jit.py`)

**MLIR Dialect Layer:**
- Purpose: Define MLIR dialects (operations, types, attributes) and implement transformation passes
- Location: `include/triton/Dialect/` (ODS/headers), `lib/Dialect/` (implementations)
- Contains:
  - `Triton` (`tt`): Core Triton IR operations (load, store, arange, reduce, etc.)
  - `TritonGPU` (`ttg`): GPU-layout-aware operations with encoding attributes
  - `Gluon`: Experimental higher-level operations including `extern_call`
  - `TritonNvidiaGPU`: NVIDIA-specific operations (tensor memory, warp group)
  - `TritonInstrument`: Instrumentation dialect for GPU monitoring
- Depends on: LLVM/MLIR upstream libraries
- Used by: Conversion passes, C++ tools (`triton-opt`), Python compiler

**Conversion Layer:**
- Purpose: Lowering between MLIR dialects and to LLVM IR
- Location: `lib/Conversion/`, `include/triton/Conversion/`, `third_party/nvidia/lib/`
- Contains:
  - `TritonToTritonGPU`: `tt` → `ttg` conversion pass
  - `TritonGPUToLLVM`: Base `ttg` → LLVM dialect conversion
  - `TritonNVIDIAGPUToLLVM`: NVIDIA-specific `ttg` → LLVM overrides (tensor memory, TMA, barriers) at `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/`
- Depends on: Dialect definitions, LLVM infrastructure
- Used by: Compiler orchestration (`CUDABackend`)

**LLVM/PTX/CUBIN Target Layer:**
- Purpose: Translate LLVM IR to PTX assembly and then to CUDA binary
- Location: `lib/Target/LLVMIR/`, `third_party/nvidia/backend/compiler.py` (make_ptx, make_cubin)
- Contains: Debug info utilities, PTX ASM emission, `ptxas` invocation
- Depends on: LLVM `TargetMachine`, NVIDIA `ptxas` toolchain
- Used by: Compiler orchestration

**C++ Bindings Layer (pybind11):**
- Purpose: Expose MLIR/LLVM/clang interop types and functions to Python
- Location: `python/src/`
- Contains: `ir.cc` (MLIR IR bindings, `set_str_attr`), `llvm.cc` (CUDA compiler bindings), `gluon_ir.cc` (Gluon op builder), `passes.cc` (MLIR pass bindings), `linear_layout.cc`, `clang_compiler.cc/.h`
- Depends on: MLIR C API, LLVM C++ API, clang libraries
- Used by: Python compiler layers

## Data Flow

### Primary Compilation Path (Triton Kernel)

1. **User writes `@triton.jit` decorated function** → `JITFunction` wrapper in `python/triton/runtime/jit.py:628`
2. **First invocation triggers compilation** → `JITFunction.run()` → `compile()` in `python/triton/compiler/compiler.py`
3. **AST → MLIR (tt dialect)** → `ast_to_ttir(fn, src, context, options, codegen_fns)` in `python/triton/compiler/code_generator.py:1662` via `CodeGenerator.visit_*()` AST traversal
4. **ttir stage** → `CUDABackend.make_ttir(mod, metadata, opt, capability)` in `third_party/nvidia/backend/compiler.py:266` — inlining, canonicalization, CSE, loop unroll
5. **ttgir stage** → `CUDABackend.make_ttgir(mod, metadata, opt, capability)` in `third_party/nvidia/backend/compiler.py:282` — `TritonToTritonGPU` conversion, coalescing, matmul acceleration, layout conversion
6. **llir stage** → `CUDABackend.make_llir(src, metadata, options, capability)` in `third_party/nvidia/backend/compiler.py:387` — `ConvertTritonGPUToLLVM` pass, MLIR→LLVM translation, LLVM optimization (O3), NVVM→LLVM conversion. If extern calls present, `_pre_compile_extern_calls()` + `link_cuda_bitcode()` injected here
7. **ptx stage** → `CUDABackend.make_ptx(src, metadata, opt, capability)` in `third_party/nvidia/backend/compiler.py:615` — LLVM→PTX assembly via `TargetMachine::addPassesToEmitFile`
8. **cubin stage** → `CUDABackend.make_cubin(src, metadata, opt, capability)` in `third_party/nvidia/backend/compiler.py:642` — PTX→CUDA binary via `ptxas`
9. **Kernel launch** → `driver.py` loads cubin and launches via CUDA driver API

### Extern CUDA C++ Interop Pipeline (`gl.call()`)

1. **Kernel calls `gl.call()`** → `python/triton/experimental/gluon/language/_core.py:774` → `call_extern()` in `_semantic.py:237`
2. **Builds MLIR op** → Creates `ttg.extern_call` MLIR op with `symbol=func, libpath=src_path`; result types inferred from `first_input.dtype/shape` + user's `result_layout=`
3. **Pre-compile extern calls** → `CUDABackend._pre_compile_extern_calls()` at `third_party/nvidia/backend/compiler.py:515` (called before `pm.run`)
   - `extract_extern_call_specs()` → MLIR operands → JSON specs (`python/src/clang_compiler.cc:756`)
   - `compile_cuda_to_module()` → invokes clang `CompilerInvocation`, `TensorTypeHelpers::InstantiateFunction()`, `CustomAstConsumer` → returns bitcode bytes + mangled function names
   - Mangled symbol names stored as module attribute
4. **Lower extern call to LLVM** → `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` — builds clang-compatible `{[N x scalar]}` struct types, args passed by pointer (`alloca + store + ptr`), post-call `extractvalue/insertvalue` back to MLIR structs
5. **Link bitcode** → `link_cuda_bitcode()` in `python/src/clang_compiler.cc` — `CloneFunctionInto` with `DifferentModule` flag, callee remapping (intrinsic decls created in dstMod), ret-type fix (`alloca+store+load` launder), `DISubprogram` strip, `alwaysinline` → O3 inlines fully

**State Management:**
- Compiled kernels cached on disk via `TRITON_CACHE_PATH` hash-based cache (`python/triton/runtime/cache.py`)
- `gl.call()` extern compilation includes `.cu` source path in cache key (`python/triton/experimental/gluon/_runtime.py`)

## Key Abstractions

**MLIR Dialect ODS:**
- Purpose: Declarative MLIR dialect definitions using TableGen
- Examples: `include/triton/Dialect/Triton/IR/TritonOps.td`, `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`, `include/triton/Dialect/Gluon/IR/GluonOps.td`
- Pattern: TableGen (`.td`) → generated C++ headers → dialect registration in `lib/Dialect/*/IR/Dialect.cpp`

**Pass Pipeline Extensions:**
- Purpose: Ordered pass pipelines registered as extensions to the MLIR pass manager
- Examples: `third_party/nvidia/backend/compiler.py` CUDABackend stage methods, `lib/Conversion/TritonGPUToLLVM/` pass registration
- Pattern: Each stage is a `compile_ir(mod, metadata) -> mod` callable stored in an ordered dict

**Layout Encoding:**
- Purpose: Represents how tensor data is distributed across GPU threads/warps/CTAs
- Examples: `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td` (BlockedEncoding, MMAEncoding, etc.)
- Pattern: MLIR attributes attached to tensor types; `LinearLayoutConversions.cpp` handles layout↔encoding bidirectional conversion

**TensorParameter / clang AST Bridge:**
- Purpose: Round-trip conversion between MLIR layout encodings and clang AST template parameters for `gl.call()`
- Examples: `python/src/clang_compiler.h` (TensorParameter struct), `python/src/gluon_ir.cc:191-280` (layoutToGluon)
- Pattern: MLIR encoding → `toLinearLayout()` → `TensorParameter{ScalarType, Shape, RegBasis, LaneBasis, WarpBasis, N_WARPS}` → `BuildTensor()` clang AST → template instantiation

**JITFunction / Cache System:**
- Purpose: Transparently compiles and caches Triton kernels behind a Python decorator
- Examples: `python/triton/runtime/jit.py:628`, `python/triton/runtime/cache.py`
- Pattern: `JITFunction.__call__()` → `DependenciesFinder` AST check → cache lookup → `compile()` if miss → `driver.launch()`

**Autotuner:**
- Purpose: Tests multiple kernel configurations (tile sizes, number of warps) and selects optimal
- Location: `python/triton/runtime/autotuner.py`
- Pattern: Decorator-based; generates ConfigSpace → benchmarks each config → selects best

## Entry Points

**`triton.jit` decorator:**
- Location: `python/triton/runtime/jit.py:935`
- Triggers: Python import-time decoration of a user function
- Responsibilities: Creates `JITFunction` wrapper, sets up dependency tracking, cache key generation

**`JITFunction.__call__`:**
- Location: `python/triton/runtime/jit.py` (JITFunction class)
- Triggers: User calls the decorated function with tensor arguments
- Responsibilities: Dependency validation, cache lookup/hit, compilation on miss, kernel launch

**`triton.compile()`:**
- Location: `python/triton/compiler/compiler.py`
- Triggers: Called by `JITFunction` on cache miss
- Responsibilities: Drives the full compilation pipeline through `CUDABackend`

**`gl.call()`:**
- Location: `python/triton/experimental/gluon/language/_core.py:774`
- Triggers: Inside a Triton Gluon kernel, calls an extern CUDA C++ function
- Responsibilities: Builds `ttg.extern_call` MLIR op, triggers in-process clang compilation and bitcode linking

**`triton-opt`:**
- Location: `bin/triton-opt.cpp`
- Triggers: CLI for pass testing and debugging
- Responsibilities: Parse MLIR, run specified passes, output transformed MLIR

**`triton-lsp`:**
- Location: `bin/triton-lsp.cpp`
- Triggers: IDE language server protocol
- Responsibilities: MLIR-aware language features for Triton MLIR files

## Architectural Constraints

- **Threading:** Python compilation runs single-threaded per kernel; async compilation supported via `_async_compile.py` for parallel kernel preloading
- **Global state:** `TRITON_CACHE_PATH` environment variable controls cache directory; `knobs.py` exposes feature flags (`python/triton/knobs.py`)
- **Circular imports:** `python/triton/language/semantic.py` ↔ `python/triton/language/core.py` (semantic layer imports core for type checks; core calls semantic for builtin implementations)
- **LLVMContext sharing:** `gl.call()` bitcode parsing reuses Triton's LLVMContext (not a temporary context) to avoid metadata leaks across `CloneFunctionInto` (`python/src/clang_compiler.cc`)
- **Build coupling:** `clang_compiler.cc` must be compiled with `-fno-rtti` because upstream clang libs are built without RTTI; permanent Clang lib linkage in root `CMakeLists.txt`
- **Python path priority:** `PYTHONPATH` must list the local source tree before the venv install (`PYTHONPATH="$(pwd)/python:$(pwd)/third_party/nvidia"`)
- **LLVM toolchain:** Uses a self-compiled LLVM at `/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/llvm-data/install`; clang is required as compiler (`CC=clang CXX=clang++`)

## Anti-Patterns

### Return Type Inference from First Argument

**What happens:** `_semantic.py:246-252` infers `gl.call()` return layout entirely from `first_input.dtype` + `first_input.shape` — ignores the actual C++ function's return type
**Why it's wrong:** Functions where return type element type, shape, or layout differs from the first argument fail silently or produce wrong results (e.g., `add_bias` in `tt_plugin.cu:117` narrows bias shape to `[1, TILE_COLS]`)
**Do this instead:** Proper C++ overload resolution + template argument deduction + return type inspection in clang Sema (see AGENTS.md "Return Type Inference (in-progress)" section); use the inferred `TensorParameter` to automatically construct `result_layout`

### C++ References as ptr in LLVM IR

**What happens:** `gl.call()` lowering passes C++ reference arguments as raw `ptr` parameters in LLVM IR (matching clang's convention), without wrapper functions
**Why it's wrong:** Tight coupling to clang's specific ABI lowering; any change in clang's reference convention breaks the lowering
**Do this instead:** Document the convention clearly; consider generating wrapper functions that manually match the ABI if robustness is needed

## Error Handling

**Strategy:** Multi-layered — Python exceptions for user-facing errors, MLIR diagnostics for compiler errors, PTXASError for toolchain failures

**Patterns:**
- `PTXASError` at `third_party/nvidia/backend/compiler.py`: raised when `ptxas` compilation fails, includes stdout/stderr
- `OutOfResources` at `python/triton/runtime/autotuner.py`: raised when autotuning can't find valid config
- `CompilationError` at `python/triton/compiler/errors.py`: raised on compilation failures with MLIR context
- MLIR reproducer printing: compiler crashes sometimes print full MLIR + `{-# ... #-}` metadata for use with `triton-opt --run-reproducer`

## Cross-Cutting Concerns

**Logging:** Python `logging` module via `python/triton/runtime/`; compiler uses `llvm::errs()` for diagnostics
**Validation:** MLIR verifiers in dialect implementations (`lib/Dialect/*/IR/`); Python-side type checking in `python/triton/language/core.py`
**Authentication:** Not applicable (offline compiler, no network services)
**Caching:** File-based cache in `TRITON_CACHE_PATH` using hash of kernel source, signature, constants, and build environment (`python/triton/runtime/cache.py`); `get_cache_invalidating_env_vars()` tracks env vars that invalidate cache

---

*Architecture analysis: 2026-07-11*
