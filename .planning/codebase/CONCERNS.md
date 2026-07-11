# Codebase Concerns

**Analysis Date:** 2026-07-11

## In-Progress / Incomplete Features

### Return Type Inference for `gl.call()` (unfinished)

**Issue:** `_semantic.py:262` infers the result type of `gl.call()` from `first_input.dtype` and `first_input.shape`. This is wrong for any function where the return type's element type, shape, or layout differs from the first argument.

**Files:** `python/triton/experimental/gluon/language/_semantic.py:246-268`, `python/src/clang_compiler.cc`, `python/src/clang_compiler.h`, `python/src/llvm.cc`

**Current state:** The C++ side has full support for the inference pipeline (all six abstractions from `clang_compiler.h`):

| Component | Status | Location |
|-----------|--------|----------|
| `TypeBuilder` | Implemented | `clang_compiler.cc:183-343` |
| `TypeInspector` | Implemented | `clang_compiler.cc:345-466` |
| `FunctionResolver` | Implemented | `clang_compiler.cc:468-568` |
| `CUDACompiler::EvaluateFunctionReturnType()` | Implemented | `clang_compiler.cc:770-782` |
| `tritonPatchExternCallResultTypes()` | Implemented | `clang_compiler.cc:950-1050` |
| Python bindings for patching | Implemented | `llvm.cc:1147-1150` |

However, the integration is only half-done:

- **`_pre_compile_extern_calls()`** (`compiler.py:553-612`) receives `ret_types` from `compile_cuda_to_module` results and stores them via `patch_extern_call_result_types` — but this only patches the MLIR module's *attributes*, it does not regenerate the MLIR op's result types. The C API `compile_cuda_to_module` itself (`clang_compiler.cc:1237-1265`) *does* pack return type info in the results tuple, so the data flows from C++ → Python.

- **The semantic layer** (`_semantic.py:250-273`) still infers result types from `first_input` before any CUDA compilation happens. The result type from CUDA is never propagated back to the IR builder. The function `call_extern` builds `result_ir_types` at line 262-266 based on `first_input.dtype`/`.shape` and never uses the CUDA-inferred types.

**Impact:** Functions where the return type's element type, shape, or layout differs from the first argument produce incorrect MLIR result types. Example: `add_bias` in `tt_plugin.cu:118-121` narrows `TILE_ROWS × TILE_COLS` input to `1 × TILE_COLS` bias but returns the full `TILE_ROWS × TILE_COLS` shape — the return type matches the *first* argument's shape, not a narrowed shape, so this specific case happens to work. But any function returning a different shape (like `reduce` in `tt_plugin.cu:127-136`, which returns `Shape<32>` from `Shape<32,32>` input) requires the user to manually supply `result_layout=`.

**Integration plan (from `AGENTS.md:112-118`):** Six steps are defined but none completed. Needs: (1) wire `EvaluateFunctionReturnType()` into `CustomAstConsumer::HandleTranslationUnit()`, (2) plumb return `TensorParameter` through `compile_cuda_to_module` result, (3) replace `_semantic.py:262-264` with inferred layout, (4) convert inferred `TensorParameter` back to `DistributedLayout` via the `layoutToGluon`/`toLinearLayout` reverse path.

**Blocking:** Blocks proper automatic layout inference for extern calls — currently every test in `test_extern_call.py` works around this by passing `result_layout=` explicitly.

---

## Fragile Areas

### LLVM Bitcode Linking in `linkBitcodeToModule()` (`clang_compiler.cc:1279-1396`)

**Files:** `python/src/clang_compiler.cc:1279-1396`

**Why fragile:** The function uses several delicate workarounds required by the LLVM `CloneFunctionInto` API and the clang→Triton LLVM module mismatch:

1. **Same `LLVMContext` requirement (line 1286):** Parsed bitcode is loaded into Triton's `LLVMContext` (passed as `ctx` parameter, not a temporary context). If the context is already modified or contains corrupt metadata, `parseIR` can fail silently or produce wrong types. No pre-validation of the context.

2. **Callee remapping (lines 1319-1336):** After `CloneFunctionInto`, all `CallInst` targets are iterated and remapped to `dstMod`'s function list. Intrinsic declarations like `llvm.lifetime.start` are not auto-created in `dstMod` by `CloneFunctionInto` — they must be explicitly `Function::Create`-ed (line 1330-1332). If a clang-generated bitcode introduces a new intrinsic not yet in `dstMod`, the `getFunction` on line 1328 returns null and a new declaration is created. If the intrinsic signature is wrong, downstream LLVM passes (including O3) may crash or miscompile.

3. **Return-type fix via alloca+store+load laundering (lines 1340-1376):** Named `%struct.Tensor` from the cloned function has a different `Type*` than the literal `{[N x scalar]}` struct in `dstFn`'s declaration — even within the *same* `LLVMContext`. The workaround iterates over `ReturnInst`s, checks if `retTy != funcRetTy`, and for each struct field performs `alloca` + `store` + `load` to launder the type. This is fragile: (a) it assumes all return type mismatches are between structurally-identical struct types, (b) it doesn't handle scalar return mismatches, (c) the alloca/load per field creates fresh SSA values that may confuse subsequent LLVM passes if not optimized away.

4. **DISubprogram fix (lines 1378-1390):** Non-distinct `DISubprogram` nodes from cloned functions are stripped and replaced with distinct ones. This prevents debug info corruption during `StripDebugInfo` (line 1395), but the fallthrough case (distinct subprograms are left untouched) has not been validated — distinct subprograms from the cloned module may reference scope/file metadata from a module that is about to be destroyed after `linkBitcodeToModule` returns.

5. **`alwaysinline` attribute forced (line 1392):** Every cloned function gets `AlwaysInline`, relying on O3 to fully inline. If a function becomes too large for inlining, LLVM may silently skip inlining, leaving a device function call in the PTX that won't be callable.

6. **StripDebugInfo global (line 1395):** Applied to `*dstMod` after all cloning — this is a sledgehammer that removes all debug info from the entire module, including non-cloned functions. Any upstream code relying on debug info after linking will see no data.

7. **No LLVM verification after linking:** Unlike the MLIR→LLVM translation step (`compiler.py:464`), there's no `verifyModule` call after `link_cuda_bitcode`. The verification is done in `compiler.py:486-487` but only after all bitcodes are linked and this verification is a Python-side check that triggers a RuntimeError. If verification succeeds, O3 is run immediately after.

**Safe modification approach:** When modifying this code: (a) add a `verifyFunction(*dstFn)` call after each clone before stripping debug info, (b) add a `verifyModule(*dstMod)` call after all cloning, (c) test with multiple simultaneously-compiled bitcode units that share internal types, (d) test with debug builds of LLVM where asserts on type mismatches are enabled.

---

### Coroutine-Based Compiler Dispatcher Pattern (global)

**Files:** `python/src/clang_compiler.cc`, `python/src/clang_compiler.h:48-116` (ExecutionContext / TargetABI / X64SysVABI)

**Why fragile:** The `CUDACompiler` operates as a coroutine using `ExecutionContext` to switch between the calling thread and the clang compile thread. Every method on `CUDACompiler` (`BuildTensor`, `BuildInts`, `LookupFunction`, `InstantiationFunction`, `EvaluateFunctionReturnType`, `EmitFinalModule`) follows the same pattern:
```cpp
// Push a task to the queue
TaskQueue.emplace([&](TensorTypeHelpers &helper, CustomAstConsumer &AstC) {
    // Do work on the clang thread
});
// Switch to the clang thread and wait for result
InvocationContext->SwitchTo(*CompileExecutionContext);
return Result;
```

**Specific fragility:**

1. **Stack-dangling references:** The lambdas pushed to `TaskQueue` capture `[&]` — references to stack-local variables in the calling method. These must complete before the method returns. If `SwitchTo` returns abnormally or the `CustomAstConsumer::HandleTranslationUnit` loop terminates early (e.g., `Running = false`), tasks may be skipped and the captured references become dangling.

2. **`__builtin_unreachable()` at `clang_compiler.cc:585`:** The `PerformCompileImpl` function ends with `__builtin_unreachable()` because control never returns from the coroutine switch. If `ExecuteAction` throws or the switch fails, this is undefined behavior — the compiler assumes this path is dead and may remove safety checks.

3. **Single-queue, no synchronization:** The `TaskQueue` is a plain `std::queue` with no mutex. It relies entirely on the coroutine handoff protocol for safety. If any task path misses a `SwitchTo`, the queue can become corrupted.

4. **X86-64-specific ABI:** The `X64SysVABI` is hardcoded with specific GPR slot indices for IP, SP, BP, Arg0, Self. This will not work on ARM64 or any non-x86 platform.

5. **Implicit `__attribute__((noinline))` on `SwitchTo` —** but the `LaunchPad` and `PerformCompileImpl` functions are not marked `noinline`, so the compiler could theoretically inline across the switch boundary, breaking the coroutine protocol.

---

### `_pre_compile_extern_calls()` Return Type Pipeline (partial)

**Files:** `third_party/nvidia/backend/compiler.py:515-612`

**Why fragile:** The pipeline compiles CUDA in-process, but the return type patching is only half-plumbed:

```python
# compiler.py:594-598
for symbol, mangled, ret_types, extr_names in results:
    mangled_names[symbol] = mangled
    if ret_types:
        return_type_map[symbol] = list(ret_types)
```

- `compile_cuda_to_module` returns `ret_types` but the C++ side actually packs them (`clang_compiler.cc:1237-1265`). It's unclear whether the path from `EvaluateFunctionReturnType()` to `compile_cuda_to_module`'s result tuple is fully wired — the `customAstConsumer::HandleTranslationUnit` does not appear to call `EvaluateFunctionReturnType()` as part of its task dispatch loop.

- `patch_extern_call_result_types` stores inferred types as a module attribute but the `ExternCallOpToLLVM.cpp` lowering pass does not read this attribute for return type construction — it reads `extern_call_mangled_names` and `extern_call_extractor_names` only (lines 14-39 and 41-69). Return types are determined from the MLIR op's result types, which were set by `_semantic.py:262-264`.

- The JSON round-trip (`_serialize_return_types` at `compiler.py:164-182`) adds overhead and potential deserialization errors.

---

## Known Bugs

### Dead Code After `return ret` (`compiler.py:505-513`)

**Files:** `third_party/nvidia/backend/compiler.py:505-513`

```python
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret                     # <-- returns here

        ret = str(llvm_mod)            # <-- unreachable
        del llvm_mod
        del context
        return ret
```

Lines 510-513 are unreachable dead code — a copy-paste artifact from a refactor. No runtime bug (dead code) but indicates rushed edits in a sensitive compilation path.

**Fix approach:** Delete lines 510-513.

---

### `f64` / `fp64` Silently Coerced to `Fp32` (`compiler.py:542`)

**Files:** `third_party/nvidia/backend/compiler.py:542`

```python
"f64": llvm.ScalarType.Fp32, "fp64": llvm.ScalarType.Fp32,
```

Double-precision types are mapped to `Fp32` silently. No error is raised. A user passing `float64` tensors to `gl.call()` will get silently truncated results.

**Fix approach:** Either: (a) implement proper `Fp64` support throughout the pipeline (requires `ScalarType::Fp64`, `getQualTypeFromScalarType` update, `Ctx.DoubleTy` support), or (b) raise a clear error at the Python API level before compilation begins.

---

### `add_bias` Stub Implementation (`tt_plugin.cu:119-121`)

**Files:** `python/test/gluon/tt_plugin.cu:118-121`

```cpp
template<typename T, uint32_t TILE_ROWS, uint32_t TILE_COLS, typename TMatLayout>
__device__ Tensor<T, Shape<TILE_ROWS, TILE_COLS>, TMatLayout> add_bias(
    const Tensor<T, Shape<TILE_ROWS, TILE_COLS>, TMatLayout>& mat,
    const Tensor<T, Shape<1, TILE_COLS>, typename TMatLayout::template Sliced<0>>& bias){
    return mat; // not implemented yet
}
```

This function returns the input unchanged. It is the example cited in `AGENTS.md` for the return-type inference limitation, which means there is no actual use case exercising the return-type inference feature.

---

### PaddedSharedLayout Visualization Unsupported (`gluon_ir.cc:1066-1067`)

**Files:** `python/src/gluon_ir.cc:1066-1067`

```cpp
if (isa<ttg::PaddedSharedEncodingAttr>(attr))
    throw py::value_error("PaddedSharedLayout cannot be visualized: "
                          "toLinearLayout not implemented");
```

`toLinearLayout` is not implemented for padded shared layouts, so any debug visualization or layout inspection on a tensor with `PaddedSharedEncodingAttr` will throw a Python exception.

---

### `__builtin_unreachable()` Without Fallback in `TypeInspector::EvaulateConstantTemplateNTTP` (`clang_compiler.cc:352-362`)

**Files:** `python/src/clang_compiler.cc:352-362`

```cpp
uint32_t TypeInspector::EvaulateConstantTemplateNTTP(const clang::TemplateArgument &Arg) {
    if (Arg.getKind() == clang::TemplateArgument::ArgKind::Integral)
        return Arg.getAsIntegral().getZExtValue();
    if (Arg.getKind() == clang::TemplateArgument::ArgKind::Expression) {
        clang::Expr::EvalResult Res;
        Arg.getAsExpr()->EvaluateAsConstantExpr(Res, Ctx);
        return Res.Val.getInt().getZExtValue();
    }
    __builtin_unreachable();
}
```

If a template argument is neither integral nor an evaluated expression (e.g., a type-dependent argument in an uninstantiated template), this is UB. Should throw an exception or return a sentinel.

Additionally, the function name has a typo: `EvaulateConstantTemplateNTTP` (should be `Evaluate`).

---

## Build / Toolchain Fragility

### Self-Compiled LLVM Dependency (hardcoded absolute path)

**Files:** `build.sh:5`, `CMakeLists.txt:32-98`, `third_party/nvidia/backend/compiler.py:529-534`

- `build.sh` sets `LLVM_SYSPATH` to `/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/llvm-data/install` — a mount-point-specific path hardcoded into the build script.
- `CMakeLists.txt:55-56` uses `-DLLVM_SYSPATH=...` as the only path to find LLVM. No fallback to system LLVM.
- `compiler.py:529-534` searches for clang resource dir at `../../../../llvm-project/install/lib/clang/23` relative to `third_party/nvidia/backend/`, falling back to `../` — both fragile relative paths.
- Symbol mismatch risk: the AGENTS.md explicitly warns against using Triton's default precompiled LLVM.

**Impact:** New developer onboarding requires building a custom LLVM at a specific path. CI/CD is impossible without reproducing this exact directory layout.

**Fix approach:** Replace hardcoded paths with environment variables (`TRITON_LLVM_PATH`, `TRITON_CLANG_RESOURCE_DIR`) with clear error messages when unset.

---

### `-fno-rtti` on `clang_compiler.cc` is Non-Standard (`CMakeLists.txt:460-461`)

**Files:** `CMakeLists.txt:460-461`

```cmake
set_source_files_properties(${PYTHON_SRC_PATH}/clang_compiler.cc
    PROPERTIES COMPILE_FLAGS "-fno-rtti")
```

Clang libraries are built without RTTI, so `clang_compiler.cc` must be compiled with `-fno-rtti`. However, this flag is applied only to `clang_compiler.cc` — other translation units in the same shared library (`llvm.cc`, `gluon_ir.cc`, `ir.cc`) do not have `-fno-rtti`. If any of those files `#include` RTTI-dependent clang headers (even transitively), link errors or runtime crashes can occur.

The AGENTS.md warns: "Clang libs are built without RTTI" and `clang_compiler.cc` is the only file compiled with `-fno-rtti` — this is correct but fragile. Any refactoring that moves clang-dependent code to other `.cc` files requires updating this build rule.

---

### Forbidden `pip install -e .` (`AGENTS.md:7`)

**Files:** `AGENTS.md:7`

Running `pip install -e .` overwrites the venv's standard triton with the local fork. The build process uses `PYTHONPATH` instead. No guard (e.g., `pip install` detection in `setup.py` or a pre-build check) prevents this. A developer following standard Python project workflow will break their environment.

---

### Hardcoded CUDA Include Path (`compiler.py:585`)

**Files:** `third_party/nvidia/backend/compiler.py:585`

```python
[cuda_inc, "/usr/local/cuda-13.1/targets/x86_64-linux/include"],
```

The CUDA include path is version-specific (cuda-13.1) and architecture-specific (x86_64-linux). Any CUDA toolkit upgrade or ARM64 host breaks compilation.

---

### Clang Resource Directory Hardcoded to Version "23" (`compiler.py:534`)

**Files:** `third_party/nvidia/backend/compiler.py:534`

```python
resource_dir = os.path.join(install_prefix, "lib", "clang", "23")
```

The clang version directory (`23`) is hardcoded. If the self-compiled LLVM is a different clang version, the resource directory won't exist and compilation will fail with opaque errors about missing headers.

---

## Security Considerations

### Arbitrary Code Execution via Extern CUDA Sources

**Files:** `python/triton/experimental/gluon/language/_core.py:774-811`, `python/src/clang_compiler.cc`, `third_party/nvidia/backend/compiler.py:515-612`

**Risk:** `gl.call(src_path, func, ...)` loads arbitrary `.cu` files from disk and compiles them in-process via clang CodeGen at JIT time. The compiled code runs as device code on the GPU.

**Current mitigation:** The `_core.py:796-801` resolves paths but does not validate file contents, sandbox the compilation, or restrict which functions from the source can be called. The resource path resolution in `compiler.py:529-534` uses relative paths from the backend directory, which could be manipulated.

**Recommendations:**
- Consider validating the `.cu` source against an allowlist of paths or a checksum
- The in-process clang compilation (`PerformCompileImpl`) runs with full clang Sema/CodeGen permissions — any clang vulnerability in the imported headers affects the host process
- Add a maximum source size limit and compilation time limit

### Hardcoded Absolute Paths Leak Machine-Specific Info

**Files:** `build.sh:5`, `third_party/nvidia/backend/compiler.py:585`

Both contain absolute paths specific to the development machine. These are committed to git (not in a `.env` file) and will not work on other machines.

---

## Performance Bottlenecks

### In-Process CUDA Compilation at JIT Time

**Files:** `python/src/clang_compiler.cc:588-693` (CUDACompiler constructor), `third_party/nvidia/backend/compiler.py:515-612`

**Problem:** Every `gl.call()` invocation triggers: (1) clang `CompilerInstance` creation, (2) parsing the `.cu` source, (3) Sema-based template argument deduction, (4) CodeGen, (5) LLVM bitcode serialization, (6) `linkBitcodeToModule` cloning. This happens **every JIT compilation**.

**Mitigation in place:** The `GluonRuntime` (`_runtime.py:25-39`) includes the `.cu` file path in the cache key hash, so compiled kernels are cached. But the compilation itself is a single-threaded bottleneck for the first time a kernel is seen.

**Observation:** The `CustomAstConsumer::HandleTranslationUnit` loop processes all tasks in a single clang thread. Parallel compilation of multiple `.cu` files is not supported — if multiple `gl.call()` instances reference different sources, they are compiled sequentially.

### String-Based Module Attribute Plumbing

**Files:** `third_party/nvidia/backend/compiler.py:601-611`

```python
error = llvm.patch_extern_call_result_types(
    mod, _json.dumps(_serialize_return_types(return_type_map)))
```

Return type information is serialized to JSON, stored as a module attribute string, and parsed back in `tritonPatchExternCallResultTypes`. This creates three serialization/deserialization round-trips (Python dict → JSON string → C++ → JSON parse → MLIR attributes). For high-frequency compilation, this adds measurable latency.

---

## Technical Debt

### Accumulated TODO/FIXME/HACK Markers

The codebase contains ~50+ TODO/FIXME/HACK/XXX markers across all layers. The most impactful:

| Location | Marker | Issue |
|----------|--------|-------|
| `llvm.cc:733` | TODO | SLP vectorizer runs with empty target machine |
| `llvm.cc:762` | TODO | Missing logging for LLVM pass insertion |
| `ExternCallOpToLLVM.cpp` | (none) | No tests, no error recovery for malformed extern_call ops |
| `WarpSpecializeUtility.cpp:64` | HACK | Barriers generated by higher-level passes |
| `WarpSpecializeUtility.cpp:286` | FIXME | Arbitrary warp group size |
| `WarpSpecializeUtility.cpp:455` | FIXME | Too many warp group partitions (abort) |
| `ScanOpToLLVM.cpp:474` | TODO | Unsupported scan layout (returns error) |
| `ViewOpToLLVM.cpp:384` | FIXME | Should compose linear layout with basis |
| `Utility.cpp:1919` | TODO | Better way needed, needs upstream fix |
| `ConvertWarpSpecializeToLLVM.cpp:267` | FIXME | Assumes warp specialization only on Blackwell |
| `TMAToLLVM.cpp:209` | TODO | ptxas bug workaround |
| `TensorMemoryToLLVM.cpp:638` | TODO | Mixed int/ptr address space handling |
| `LoadStoreOpToLLVM.cpp:1080` | XXX(Keren) | Always assume other=0 |
| `code_generator.py:454` | XXX | Hack to get insertion point location |
| `code_generator.py:335` | TODO | Illegal names for non-kernel functions with constexprs |
| `compiler.py:104` | TODO | Handle non-"a" SMs |
| `compiler.py:294` | TODO | Move PlanCTAPass to front of CoalescePass |
| `language/core.py:1304` | TODO | Remove (unclear what) |
| `language/__init__.py:328` | FIXME | Last dim stride should be constexpr(1) |

### Incomplete / Stub Error Handling

- `ExternCallOpToLLVM.cpp:129-132`: Error emits a diagnostic but the lowering pass may continue, producing an invalid LLVM module that fails at verification time with a cryptic error.
- `clang_compiler.cc:239`: `assert(!N_BASES || vecs.size() % N_BASES == 0)` — crashes in release builds if layout basis vectors don't match expected count.
- `compiler.py:696-719`: ptxas failure handling is good (catches SIGSEGV and prints reproducer) but doesn't save the failing PTX to a file for offline debugging.

### Large File Risk

- `python/src/clang_compiler.cc` — **1,396 lines** in a single file. Contains CUDACompiler, TypeInspector, TypeBuilder, FunctionResolver, MLIR spec extraction, and bitcode linking all in one translation unit. Should be split into separate files before it grows further.

### Misnamed Function

- `clang_compiler.cc:352`: `TypeInspector::EvaulateConstantTemplateNTTP` — typo: should be `EvaluateConstantTemplateNTTP` (two `a`s where one `l` should be).

---

## Test Coverage Gaps

### Extern CUDA Call (`gl.call`) Testing

**Files:** `python/test/gluon/test_extern_call.py`

**What's tested (4 tests):**
1. `test_elementwise_add` — same-layout elementwise operation
2. `test_intra_warp_add_sibling` — intra-warp shuffle operation
3. `test_reduce_different_shape` — reduction across one dimension
4. `test_split_add_tuple` — multi-return (tuple) operation

**What's NOT tested:**
- Functions where return type **element type** differs from input (e.g., float16→float32)
- Functions where return type **shape** differs from first input (the exact limitation described in AGENTS.md)
- Functions with `fast_math` flag
- Functions with `assert_no_conv` flag
- Error cases: missing source file, missing function, type mismatch
- Multiple `.cu` files in a single kernel (different libpaths)
- Concurrent compilation of multiple `gl.call` instances
- `ScalarType::Int32`, `Int64`, `Bf16`, `Fp8` — all tests use `Fp32`
- `use_fast_math=True`
- Non-trivial layouts (all tests use simple 1D blocked layouts)
- PGO / profile-guided compilation with extern calls

**Risk:** A change to the return-type convention (e.g., the planned inference feature) could silently break all four existing tests and go undetected until runtime errors.

### `clang_compiler.cc` Unit Tests

**No unit tests exist** for any of the following:
- `TypeInspector::DispatchTypeParsing` with invalid/malformed clang AST types
- `FunctionResolver::LookupFunction` with ambiguous overloads
- `linkBitcodeToModule` with edge-case LLVM bitcode (empty functions, recursive calls, varargs)
- `tritonExtractExternCallSpecs` with nested layouts
- `tritonPatchExternCallResultTypes` with mismatched result counts
- Coroutine protocol edge cases (queue overflow, early termination)

---

## Scaling Limits

### Single-Threaded CUDA Compilation

**Files:** `python/src/clang_compiler.cc:401-416`

The `CustomAstConsumer::HandleTranslationUnit` processes all compilation tasks sequentially in a single thread. The coroutine protocol (`ExecutionContext::SwitchTo`) is designed for single-threaded cooperative multitasking, not parallelism. If a kernel uses 10 different `gl.call()` invocations referencing 10 different `.cu` files, compilation is serial — each file is parsed and codegen'd one at a time.

**Scaling path:** The `CUDACompiler` would need to be instantiated per-source-file with per-thread `LLVMContext` instances. The `linkBitcodeToModule` call would need thread-safe access to `dstMod`.

### Cache Key Sensitivity

**Files:** `python/triton/experimental/gluon/_runtime.py:25-39`

The cache key includes the `.cu` file path *as a string*. If the same source file is at different absolute paths (e.g., due to symlinks or different working directories), the cache key differs and the kernel is recompiled. Using a content hash of the source file would be more robust.

---

*Concerns audit: 2026-07-11*
