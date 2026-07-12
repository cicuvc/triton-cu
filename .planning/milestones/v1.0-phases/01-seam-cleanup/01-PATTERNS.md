# Phase 01: Seam & Cleanup - Pattern Map

**Mapped:** 2026-07-11
**Files analyzed:** 7 modified (0 new files)
**Analogs found:** 7 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `third_party/nvidia/backend/compiler.py` | backend-compiler | request-response + batch-compile | `compiler.py:246-254` (get_codegen_implementation), `compiler.py:515-612` (_pre_compile_extern_calls) | exact — same file, same methods |
| `python/triton/experimental/gluon/language/_semantic.py` | semantic-frontend | request-response (IR-build) | `semantic.py:835-837` (convert_custom_types consumption), `semantic.py:1477-1479` (min_dot_size consumption) | role-match — same semantic layer, same codegen_fns dispatch |
| `python/triton/experimental/gluon/_runtime.py` | ir-source | request-response (make_ir entry) | `_runtime.py:47-69` (existing make_ir) | exact — same file, same function |
| `python/triton/compiler/compiler.py` | compiler-orchestration | request-response (stage dispatch) | `compiler.py:304-307` (hook passthrough) | exact — same file, same lines |
| `python/triton/compiler/code_generator.py` | code-generator | request-response (builder setup) | `code_generator.py:312` (builder.codegen_fns assignment) | exact — same file, same lines |
| `python/src/clang_compiler.cc` / `.h` | c++-backend | event-driven (coroutine dispatch) | `clang_compiler.cc:696-795` (CUDACompiler method pattern), `clang_compiler.cc:1145-1277` (tritonCompileCuda lifecycle) | exact — same file, same class |
| `python/src/llvm.cc` | python-bindings | request-response (C++→Python) | `llvm.cc:1107-1134` (compile_cuda_to_module binding) | exact — same file, same patterns |

## Pattern Assignments

---

### 1. `third_party/nvidia/backend/compiler.py` — `get_codegen_implementation` (adding the inference hook)

**Analog:** `compiler.py:246-254` — existing `convert_custom_types` + `min_dot_size` hook exposure

**Imports pattern** (lines 1-4):
```python
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, nvidia
from triton import knobs
from triton.runtime.errors import PTXASError
```

**Core codegen_fns dict pattern** (lines 246-254):
```python
def get_codegen_implementation(self, options):
    import triton.language.extra.cuda as cuda
    capability = int(self._parse_arch(options.arch))
    codegen_fns = {
        "convert_custom_types":
        cuda.convert_custom_float8_sm80 if capability >= 80 else cuda.convert_custom_float8_sm70, "min_dot_size":
        min_dot_size(self.target)
    }
    return codegen_fns
```

**Key insight for new hook:** The inference object (a stateful object with `.infer_result()` and `.compile_bitcode()` methods) is added as a new key in this dict. Since `get_codegen_implementation` has access to `options` (for `sm`/`resource_dir`/include paths), the hook object can be constructed here. The dict key name is at the agent's discretion (suggested: `"infer_extern_call_result"`).

**Pattern to copy:** Add a new key-value pair in the `codegen_fns` dict:

```python
codegen_fns = {
    "convert_custom_types": ...,
    "min_dot_size": ...,
    "infer_extern_call_result": SomeInferenceClass(self, options),  # NEW
}
```

---

### 2. `python/triton/language/semantic.py` — how `codegen_fns` is consumed (pattern for any future consumer)

**Analog:** Lines 835-837 (`convert_custom_types`), lines 1477-1479 (`min_dot_size`)

**Core codegen_fns consumption pattern** (lines 835-837):
```python
if (src_sca_ty.is_fp8e4b15() or dst_sca_ty.is_fp8e4b15()):
    assert self.builder.codegen_fns.get(
        "convert_custom_types") is not None, "target doesn't provide conversion for this type."
    return self.builder.codegen_fns["convert_custom_types"](input, dst_ty, fp_downcast_rounding, _semantic=self)
```

**Core codegen_fns consumption pattern** (lines 1477-1479):
```python
assert self.builder.codegen_fns.get(
    "min_dot_size") is not None, "target doesn't provide lower shape bounds for dot."
min_dot_size = self.builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
```

**Pattern to copy (Phase 2, for reference only):**
```python
infer = self.builder.codegen_fns.get("infer_extern_call_result")
if infer is None:
    raise RuntimeError("gl.call() extern CUDA calls require the CUDA backend.")
returned = infer.infer_result(func, arg_params, use_fast_math)
```

**Note for Phase 1:** `call_extern` is UNCHANGED in Phase 1 except the f64 guard. This pattern is documented here so the planner knows exactly where the hook will eventually be consumed.

---

### 3. `python/triton/experimental/gluon/language/_semantic.py` — `call_extern` f64 early guard

**Analog:** `_semantic.py:250-273` — the existing `call_extern` function body

**Current call_extern** (lines 250-273):
```python
def call_extern(self, src_path, func, args, result_layouts, assert_no_conv=False, use_fast_math=False):
    _check(isinstance(src_path, str), lambda: f"expected 'src_path' to be a str but got {type(src_path)!r}")
    _check(isinstance(func, str), lambda: f"expected 'func' to be a str but got {type(func)!r}")
    for a in args:
        _check(isinstance(a, ttgl.tensor),
               lambda: f"all arguments must be tensors but got {type(a)!r}")
    _check(isinstance(result_layouts, list),
           lambda: f"result_layouts must be a list but got {type(result_layouts)!r}")

    first_input = args[0]
    result_types = []
    for lo in result_layouts:
        result_shape = _compute_result_shape(first_input.shape, lo)
        result_types.append(
            ttgl.distributed_type(first_input.dtype, result_shape, lo))

    result_ir_types = [rt.to_ir(self.builder) for rt in result_types]
    arg_handles = [a.handle for a in args]
    result_handles = self.builder.create_extern_call(
        str(src_path), func, arg_handles, result_ir_types,
        assert_no_conv, use_fast_math)

    results = [ttgl.tensor(h, rt) for h, rt in zip(result_handles, result_types)]
    return results[0] if len(results) == 1 else tuple(results)
```

**f64 guard pattern** (D-09 — backend-agnostic dtype-string check):

Insert BEFORE the `first_input = args[0]` line (after the arg validation block). Pattern:
```python
# Backend-agnostic f64 guard — raise before building IR.
_f64_err = "gl.call() does not support float64; full Fp64 support is out of scope (see FP64-01)"
for a in args:
    if hasattr(a, 'dtype') and str(a.dtype) in ("fp64", "f64", "float64"):
        raise NotImplementedError(_f64_err)
```

This is a pure dtype-string check — no CUDA import, preserves layering. The guard also applies to `float64` (the canonical Triton dtype name for f64, if it exists). The agent may adjust the exact strings checked.

---

### 4. `third_party/nvidia/backend/compiler.py` — `_pre_compile_extern_calls` (resuming the suspended CUDACompiler)

**Analog:** `compiler.py:515-612` — the existing `_pre_compile_extern_calls` method

**Key by_libpath grouping pattern** (lines 547-551):
```python
# Group by libpath
by_libpath = {}
for spec in specs_list:
    libpath = spec["libpath"]
    by_libpath.setdefault(libpath, []).append(spec)
```

**dtype_to_scalar pattern** (lines 538-545) — BEFORE (current, with silent coercion):
```python
dtype_to_scalar = {
    "f32": llvm.ScalarType.Fp32, "fp32": llvm.ScalarType.Fp32,
    "f16": llvm.ScalarType.Fp16, "fp16": llvm.ScalarType.Fp16,
    "bf16": llvm.ScalarType.Bf16,
    "f64": llvm.ScalarType.Fp32, "fp64": llvm.ScalarType.Fp32,
    "i32": llvm.ScalarType.Int32, "s32": llvm.ScalarType.Int32,
    "i64": llvm.ScalarType.Int64, "s64": llvm.ScalarType.Int64,
}
```

**dtype_to_scalar pattern** — AFTER (D-09, BUG-02 backend backstop):
```python
dtype_to_scalar = {
    "f32": llvm.ScalarType.Fp32, "fp32": llvm.ScalarType.Fp32,
    "f16": llvm.ScalarType.Fp16, "fp16": llvm.ScalarType.Fp16,
    "bf16": llvm.ScalarType.Bf16,
    "i32": llvm.ScalarType.Int32, "s32": llvm.ScalarType.Int32,
    "i64": llvm.ScalarType.Int64, "s64": llvm.ScalarType.Int64,
}


def _scalar_type_for(dtype_str):
    st = dtype_to_scalar.get(dtype_str)
    if st is None:
        if dtype_str in ("f64", "fp64", "float64"):
            raise NotImplementedError(
                "gl.call() does not support float64; full Fp64 support is out of scope (see FP64-01)")
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return st
```

Then replace `dtype_to_scalar.get(inp["dtype"], llvm.ScalarType.Fp32)` at line 572 with `_scalar_type_for(inp["dtype"])`.

**Per-libpath suspended compiler stash** (D-05, D-06 — new pattern to add):

In the `for libpath, specs in by_libpath.items()` loop, before the `with open(libpath)` block, pull the suspended compiler from metadata:

```python
# Pull suspended CUDACompiler from metadata (stashed by make_ir at semantic time).
# Key format: "__extern_cuda_compiler__" + resolved libpath.
suspended_compilers = metadata.get("__extern_cuda_compilers__", {})

for libpath, specs in by_libpath.items():
    compiler = suspended_compilers.get(libpath)
    if compiler is not None:
        # Resume: call compile_bitcode on the suspended compiler instead
        # of the old compile_cuda_to_module path.  Returns (bitcode, names).
        bitcode_list, mangled_names_batch, extractor_names_batch = (
            compiler.compile_bitcode(specs))
        # ... store results ...
        compiled_bitcodes.append(bitcode_list)
        mangled_names.update(mangled_names_batch)
        extractor_names.update(extractor_names_batch)
        continue  # Skip the old compile_cuda_to_module path

    # Fallback: existing compile_cuda_to_module path for the
    # (not-yet-exposed) case.  This exists during migration.
    with open(libpath) as f:
        source = f.read()
    # ... existing compile_cuda_to_module code ...
```

**metadata stash pattern** (lines 609-612 — existing):
```python
metadata["extern_call_bitcodes"] = compiled_bitcodes
metadata["extern_call_mangled"] = mangled_names
metadata["extern_call_extractor_names"] = extractor_names
```

**New metadata stash** (same block, add):
```python
# New: signal single-parse count for assertion (D-07).
metadata["__extern_cuda_parse_count__"] = len(compiled_bitcodes)
```

These ride the same `metadata` channel as the existing entries.

---

### 5. `third_party/nvidia/backend/compiler.py` — Dead code removal (BUG-01)

**Analog:** Lines 505-513 — the existing code block with the unreachable duplicate

**Current code** (lines 505-513):
```python
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

        ret = str(llvm_mod)          # UNREACHABLE (D-10)
        del llvm_mod                 # UNREACHABLE
        del context                  # UNREACHABLE
        return ret                   # UNREACHABLE
```

**After cleanup:** Delete lines 510-513. Only `505: return ret` remains. The `return` at line 508 already returns unconditionally, so lines 510-513 are dead code.

---

### 6. `third_party/nvidia/backend/compiler.py` — `make_llir` (metadata consumption context)

**Analog:** Lines 436-484 — how `_pre_compile_extern_calls` result is consumed downstream

**Existing consumption pattern** (lines 436-444):
```python
has_extern_calls = self._pre_compile_extern_calls(
    mod, metadata, capability, context)
if has_extern_calls:
    mod.set_str_attr("ttg.extern_call_mangled_names",
                     _json.dumps(metadata["extern_call_mangled"]))
    if metadata.get("extern_call_extractor_names"):
        mod.set_str_attr("ttg.extern_call_extractor_names",
                         _json.dumps(metadata["extern_call_extractor_names"]))
```

**Existing bitcode linking** (lines 481-484):
```python
if has_extern_calls:
    bitcodes = metadata.get("extern_call_bitcodes", [])
    for bc in bitcodes:
        llvm.link_cuda_bitcode(llvm_mod, bytes(bc), context)
```

**New assertion pattern** (D-07 — parse count assertion):

After `has_extern_calls` block, before `pm.run`:
```python
if has_extern_calls:
    parse_count = metadata.get("__extern_cuda_parse_count__", 0)
    distinct_cu = len(set(spec["libpath"] for spec in specs_list)) if has_extern_calls else 0
    assert parse_count == distinct_cu, \
        f"extern CUDA parse count mismatch: {parse_count} parses for {distinct_cu} .cu files"
```

This hard assertion catches any redundant parses — a regression guard, not prose (roadmap SC3).

---

### 7. `python/triton/experimental/gluon/_runtime.py` — `make_ir` (creating + suspending + stashing CUDACompiler)

**Analog:** `_runtime.py:47-69` — the existing `make_ir` function

**Current make_ir** (lines 47-69):
```python
def make_ir(self, target, options, codegen_fns, module_map, context):
    from triton.compiler.compiler import make_backend
    from triton.compiler.code_generator import ast_to_ttir

    builder = ir.builder(context)
    module = builder.create_module()

    backend = make_backend(target)
    target = backend.get_target_name(options)

    module.set_attr("ttg.target", builder.get_string_attr(target))
    module.set_attr("ttg.num-warps", builder.get_int32_attr(options.num_warps))
    module.set_attr("ttg.num-ctas", builder.get_int32_attr(options.num_ctas))
    module.set_attr("ttg.threads-per-warp", builder.get_int32_attr(options.warp_size))

    is_cuda = options.backend_name == "cuda"
    if is_cuda and options.maxnreg is not None:
        module.set_attr("ttg.maxnreg", builder.get_int32_attr(options.maxnreg))

    module = ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                         module_map=module_map, module=module)
    return module
```

**New pattern** (D-05, D-06 — create+suspend+stash CUDACompiler before `ast_to_ttir`):

The suspended CUDACompiler is created at semantic time, before the extern_call MLIR ops are built. Insert after `module.set_attr("ttg.maxnreg", ...)` and before the `ast_to_ttir` call:

```python
is_cuda = options.backend_name == "cuda"
if is_cuda and options.maxnreg is not None:
    module.set_attr("ttg.maxnreg", builder.get_int32_attr(options.maxnreg))

# NEW: Create suspended CUDACompiler(s) for each .cu file referenced
# by gl.call() sites. Stash in metadata so _pre_compile_extern_calls
# can resume them at the llir stage.
suspended_compilers = {}
# Scan for gl.call patterns (same regex as parse_options in this file).
import re
from pathlib import Path
try:
    if hasattr(self.fn, 'raw_src'):
        source = ''.join(self.fn.raw_src)
    else:
        source = ""
    cu_paths = set()
    for m in re.finditer(r'gl\s*\.\s*call\s*\(\s*["\']([^"\']+\.cu)["\']', source):
        cu_path = Path(m.group(1))
        if not cu_path.is_absolute():
            cu_path = Path.cwd() / cu_path
        cu_path = cu_path.resolve()
        if cu_path.exists():
            cu_paths.add(str(cu_path))
except Exception:
    cu_paths = set()

for cu_path in cu_paths:
    with open(cu_path) as f:
        cu_source = f.read()
    # Create CUDACompiler and park it BEFORE HandleTranslationUnit completes.
    # The inference hook (on codegen_fns) uses this suspended compiler.
    # Construction details at agent discretion: sm, resource_dir, include paths
    # come from get_codegen_implementation or options.
    pass  # Actual construction: create CUDACompiler, call PerformParse,
          # then park (do NOT call EmitFinalModule). Store the suspended
          # compiler object. See exec notes below.

module = ast_to_ttir(self.fn, self, ...)
```

**Critical exec-level note:** The PYTHONPATH mechanism (not `pip install -e`) and the LLVM/Clang shared-library linkage means the CUDACompiler can only be constructed when `llvm` (the C extension) is importable. `make_ir` runs after `backend.load_dialects(context)` so this is satisfied.

**metadata stash** (D-05): The `metadata` dict is passed to `ast_to_ttir` which builds the MLIR module. The suspended compiler stash must go into `metadata` BEFORE it reaches `_pre_compile_extern_calls` at the llir stage. Since `make_ir` returns a `module` (not metadata), the stash must be attached to the module as an attribute (like how `extern_call_mangled_names` is set) OR threaded through a separate channel.

**Resolution:** The planner should decide: (a) store the suspended compilers on the module as a private attribute, (b) use the `metadata` dict that flows through `compile_ir` callables, or (c) add a `metadata` parameter to `make_ir`. Option (b) is most natural for the existing stage architecture — each stage is `compile_ir(mod, metadata) -> mod`, so `make_ir` should accept and mutate `metadata`.

---

### 8. `python/triton/compiler/compiler.py` — Hook passthrough (no modification needed)

**Analog:** Lines 304-307 — existing passthrough

**Existing pattern** (lines 304-307):
```python
codegen_fns = backend.get_codegen_implementation(options)
module_map = backend.get_module_map()
try:
    module = src.make_ir(target, options, codegen_fns, module_map, context)
except Exception as e:
    filter_traceback(e)
    raise
```

**No code change needed.** The inference hook added to `codegen_fns` in `get_codegen_implementation` automatically threads through to `src.make_ir(...)` via the existing `codegen_fns` parameter. The only consideration is whether `metadata` should also be passed — see the resolution note above in section 7.

---

### 9. `python/triton/compiler/code_generator.py` — `self.builder.codegen_fns = codegen_fns`

**Analog:** Line 312 — existing propagation

**Existing pattern** (lines 309-312):
```python
# dict of functions provided by the backend. Below are the list of possible functions:
# Convert custom types not natively supported on HW.
# convert_custom_types(input_tensor, dtype, fp_downcast_rounding=None, _builder=None)
self.builder.codegen_fns = codegen_fns
```

**Existing ast_to_ttir signature** (lines 1662-1700):
```python
def ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
    ...
    generator = CodeGenerator(context, prototype, gscope=..., ..., codegen_fns=codegen_fns,
                               module_map=module_map, module=module, is_gluon=fn.is_gluon())
```

**No code change needed.** The new hook key travels with `codegen_fns` through this existing propagation chain. The semantic layer accesses it via `self.builder.codegen_fns["infer_extern_call_result"]`.

---

### 10. `python/src/clang_compiler.cc` / `.h` — CUDACompiler coroutine suspend/resume

**Analog:** `clang_compiler.cc:696-795` (CUDACompiler method pattern), `clang_compiler.cc:1145-1277` (tritonCompileCuda lifecycle)

**Core coroutine dispatch pattern** (every CUDACompiler method follows this — example from lines 696-728):
```cpp
clang::QualType
CUDACompiler::BuildTensor(const TensorParameter &Param) {
  clang::QualType Result;
  TaskQueue.emplace([&](TensorTypeHelpers &helper,      // [&] capture!
                        CustomAstConsumer &) {
    auto Shape = helper.Builder.buildShape(Param.Shape);
    // ... work on clang thread ...
    Result = ...;  // Writes through [&] capture
  });
  InvocationContext->SwitchTo(*CompileExecutionContext);
  return Result;  // [&] capture must outlive the SwitchTo!
}
```

**Critical [&] capture protocol** (from CONCERNS.md §Coroutine-Based Compiler Dispatcher Pattern):
- `TaskQueue.emplace([&](...) { ... })` captures by reference
- `SwitchTo` yields control to the clang thread; the `[&]` captures (locals on the calling frame) MUST remain alive
- The method returns a local (`Result`) that was written by the clang thread's lambda
- This works because `SwitchTo` is synchronous from the caller's perspective — the clang thread runs the lambda and pushes back to the calling thread before the method returns

**New suspend pattern** (D-03 — park before HandleTranslationUnit completes):

The key insight is that `PerformParse` (line 683) starts the clang parse loop, and `HandleTranslationUnit` (lines 401-416) is the coroutine loop inside the parse:

```cpp
// Existing HandleTranslationUnit coroutine loop (lines 401-416):
void HandleTranslationUnit(clang::ASTContext &Ctx) override {
    TensorTypeHelpers helper(Ctx, ci.getSema());
    Compiler.CompileExecutionContext->SwitchTo(
        *Compiler.InvocationContext.get());

    while (Running) {
      while (Compiler.TaskQueue.size()) {
        Compiler.TaskQueue.front()(helper, *this);
        Compiler.TaskQueue.pop();
      }
      if (Running)
        Compiler.CompileExecutionContext->SwitchTo(
            *Compiler.InvocationContext.get());
    }
}
```

**Suspend mechanism:** The `Running` flag already controls whether the coroutine yields or exits. To suspend before `CodeGen->HandleTranslationUnit` (which finalizes LLVMModule and would tear down ASTContext):

1. After `PerformParse` returns (semantic time), the coroutine is parked inside `HandleTranslationUnit` waiting for tasks — the `ASTContext` is alive, `Running` is `true`
2. Inference calls (`BuildTensor`, `LookupFunction`, `EvaluateFunctionReturnType`, `InstantiationFunction`) queue tasks and switch into the clang thread; each completes and the coroutine parks again
3. At emit time (llir stage), a new method is called that sets `Running = false` AFTER queuing the `EmitFinalModule`-like task, which runs `CodeGen->HandleTranslationUnit(...)` and releases the module

**Pattern for the suspend between parse and emit:**

Instead of the existing `tritonCompileCuda` which does parse → infer → codegen → emit in one shot (lines 1164-1277), split into:

1. **Semantic stage:** `CreateSuspendedCompiler(source, sm, resource_dir, include_paths)` → returns a (parse-counted, suspended) compiler handle to Python
2. **Inference (can be called per request):** `compiler.lookup_and_infer(symbol, arg_types)` → returns `TensorParameter` results
3. **llir stage:** `compiler.emit_bitcode()` → returns `(bitcode, results)` — resumes the coroutine for the final emit

The `tritonCompileCuda` function at lines 1145-1277 is the monolithic analog — Phase 1 splits Phase 1 (parse+park) from Phase 3 (emit) without changing Phase 2 (inference calls).

**EmitFinalModule pattern** (lines 784-795) — the final task that completes parsing:
```cpp
std::unique_ptr<llvm::Module> CUDACompiler::EmitFinalModule() {
  std::unique_ptr<llvm::Module> Result;
  TaskQueue.emplace([&](TensorTypeHelpers &helper,
                        CustomAstConsumer &AstC) {
    AstC.CodeGen->HandleTranslationUnit(
        AstC.ci.getASTContext());
    AstC.Running = false;
    Result = AstC.CodeGen->ReleaseModule();
  });
  InvocationContext->SwitchTo(*CompileExecutionContext);
  return Result;
}
```

This is the pattern for "resume and consume" — Phase 1 ensures this can be called separately from the inference calls.

**Header struct** (lines 316-345) — the CUDACompiler already has all needed members:
```cpp
struct CUDACompiler {
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> Vfs;
  std::unique_ptr<llvm::WritableMemoryBuffer> SourceBuffer;
  std::unique_ptr<clang::CompilerInstance> CI;

  std::unique_ptr<ExecutionContext> CompileExecutionContext;
  std::unique_ptr<ExecutionContext> InvocationContext;

  std::queue<std::function<void(TensorTypeHelpers &,
                                CustomAstConsumer &)>>
      TaskQueue;

  // Existing methods:
  static void PerformCompileImpl(uint64_t Arg0, ExecutionContext &ExecCtx);
  CUDACompiler(...);
  void PerformParse(llvm::LLVMContext &Context, const llvm::StringRef &ModuleName);
  clang::QualType BuildTensor(const TensorParameter &);
  clang::QualType BuildInts(uint32_t N);
  clang::FunctionDecl *LookupFunction(...);
  std::variant<...> EvaluateFunctionReturnType(clang::FunctionDecl *FD);
  llvm::Function *InstantiationFunction(clang::FunctionDecl *);
  std::unique_ptr<llvm::Module> EmitFinalModule();
};
```

**Parse counter** (D-07): Add a `static int sParseCount = 0` counter inside the `PerformParse` path, increment it on each actual `CompilerInstance::ExecuteAction`. The Python side calls `get_parse_count()` through bindings and asserts it equals the number of distinct `.cu` files.

---

### 11. `python/src/llvm.cc` — Python bindings for the new plumbing

**Analog:** Lines 1107-1134 (`compile_cuda_to_module` binding)

**Existing compile_cuda_to_module binding** (lines 1107-1134):
```cpp
m.def("compile_cuda_to_module",
       [](llvm::LLVMContext &ctx, const std::string &source,
          const std::string &sm, const std::string &resourceDir,
          const std::vector<std::string> &includePaths,
          const std::vector<CudaFuncRequest> &requests) -> py::tuple {
         auto [bitcode, error, results] =
             tritonCompileCuda(ctx, source, sm, resourceDir, includePaths,
                               requests);
         if (error.empty()) {
           py::list pyResults;
           for (auto &r : results) {
             py::list pyRetTypes;
             for (auto &tp : r.ReturnTypes)
               pyRetTypes.append(py::cast(tp));
             py::list pyExtrNames;
             for (auto &n : r.ExtractorMangledNames)
               pyExtrNames.append(py::str(n));
             pyResults.append(py::make_tuple(r.Symbol, r.MangledName,
                                              pyRetTypes, pyExtrNames));
           }
           return py::make_tuple(true, py::bytes(bitcode), py::str(""),
                                 pyResults);
         } else {
           return py::make_tuple(false, py::none(), py::str(error),
                                 py::list());
         }
       },
       py::arg("ctx"), py::arg("source"), py::arg("sm"),
       py::arg("resource_dir"), py::arg("include_paths"),
       py::arg("requests"));
```

**ScalarType enum binding** (lines 946-953) — used by both old and new path:
```cpp
py::enum_<ScalarType>(m, "ScalarType")
    .value("Int32", ScalarType::Int32)
    .value("Int64", ScalarType::Int64)
    .value("Fp32", ScalarType::Fp32)
    .value("Fp16", ScalarType::Fp16)
    .value("Bf16", ScalarType::Bf16)
    .value("Fp8e4m3", ScalarType::Fp8e4m3)
    .value("Fp8e5m2", ScalarType::Fp8e5m2);
```

**TensorParameter binding** (lines 955-1003) — with all fields (type, shape, layout_shape, reg_basis, lane_basis, warp_basis, n_warps)

**CudaFuncRequest binding** (lines 1005-1009):
```cpp
py::class_<CudaFuncRequest>(m, "CudaFuncRequest")
    .def(py::init<>())
    .def_readwrite("symbol", &CudaFuncRequest::Symbol)
    .def_readwrite("param_types", &CudaFuncRequest::ParamTypes)
    .def_readwrite("use_fast_math", &CudaFuncRequest::UseFastMath);
```

**New bindings needed** (Phase 1):

The planner should determine the exact binding surface, but the patterns are:

1. **Create suspended compiler** — a new binding that constructs a `CUDACompiler`, calls `PerformParse`, and returns a Python-callable handle (without calling `EmitFinalModule`):
   ```cpp
   m.def("create_suspended_cuda_compiler", ...)  // returns a py::capsule or custom Python object
   ```

2. **Resume and emit** — a method on the suspended compiler handle that completes the parse and returns bitcode:
   ```cpp
   // On the suspended compiler Python object:
   .def("emit_bitcode", [](...) { ... })  // calls EmitFinalModule
   ```

3. **Parse counter** (D-07):
   ```cpp
   m.def("get_extern_cuda_parse_count", []() -> int { return sParseCount; });
   ```

All bindings follow the same `py::arg(...)` pattern as the existing code.

---

## Shared Patterns

### Coroutine `[&]` capture + `SwitchTo` dispatch protocol
**Source:** `clang_compiler.cc:696-795`
**Apply to:** All new CUDACompiler methods that operate on the clang thread
**Pattern:**
```cpp
ReturnType CUDACompiler::MethodName(Args...) {
  ReturnType Result;
  TaskQueue.emplace([&](TensorTypeHelpers &helper, CustomAstConsumer &AstC) {
    // Work on ASTContext / Sema / CodeGen — written through [&] capture.
    Result = ...;
  });
  InvocationContext->SwitchTo(*CompileExecutionContext);
  return Result;  // [&] capture of Result must outlive SwitchTo above.
}
```
**Critical rule:** All captures are by reference (`[&]`). The captured locals (including `Result`) must remain alive across the `SwitchTo` — which they do because `SwitchTo` is synchronous from the caller's perspective.

### metadata dict threading between compile stages
**Source:** `compiler.py:609-612` (write), `compiler.py:436-484` (read)
**Apply to:** All stage transitions (semantic → llir)
**Pattern:**
```python
# Writer side (_pre_compile_extern_calls):
metadata["extern_call_bitcodes"] = compiled_bitcodes
metadata["extern_call_mangled"] = mangled_names

# Reader side (make_llir):
if has_extern_calls:
    bitcodes = metadata.get("extern_call_bitcodes", [])
    for bc in bitcodes:
        llvm.link_cuda_bitcode(llvm_mod, bytes(bc), context)
```

### codegen_fns dict key-based dispatch
**Source:** `compiler.py:246-254` (backend registration), `semantic.py:835-837` (frontend consumption)
**Apply to:** All backend→frontend hook communication
**Pattern:**
```python
# Backend registration:
codegen_fns = {"key_name": backend_implementation, ...}

# Frontend consumption:
impl = self.builder.codegen_fns.get("key_name")
assert impl is not None, "backend doesn't support this feature"
result = impl(args)
```

### Error handling — explicit raise with clear message
**Source:** `compiler.py:588-590` (CUDA compilation failure), `_semantic.py:251-257` (argument validation)
**Apply to:** f64 guard, absent-hook guard
**Pattern:**
```python
# Validation with clear error:
_check(isinstance(...), lambda: f"expected ... but got {type(a)!r}")

# Backend error:
raise RuntimeError(f"In-process CUDA compilation failed for {libpath}: {error}")

# For Phase 1 f64 guard:
raise NotImplementedError("gl.call() does not support float64; ...")
```

---

## No Analog Found

All files have exact or role-matched analogs within the same codebase. No files require falling back to RESEARCH.md patterns.

---

## Metadata

**Analog search scope:** `third_party/nvidia/backend/compiler.py`, `python/triton/compiler/compiler.py`, `python/triton/compiler/code_generator.py`, `python/triton/language/semantic.py`, `python/triton/experimental/gluon/language/_semantic.py`, `python/triton/experimental/gluon/_runtime.py`, `python/src/clang_compiler.cc`, `python/src/clang_compiler.h`, `python/src/llvm.cc`, `.planning/codebase/CONCERNS.md`
**Files scanned:** 9 source files + 2 planning docs
**Pattern extraction date:** 2026-07-11
