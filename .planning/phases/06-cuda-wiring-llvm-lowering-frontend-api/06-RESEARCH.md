# Phase 06: CUDA Wiring + LLVM Lowering + Frontend API - Research

**Researched:** 2026-07-16
**Domain:** CUDA compilation wiring, MLIR→LLVM lowering, Gluon frontend API
**Confidence:** HIGH

## Summary

Phase 6 routes shared-memory args end-to-end from `gl.call()` through CUDA compilation to `ptr addrspace(3)` LLVM IR emission. The wiring (SHWIRE-01) extends `_pre_compile_extern_calls()` to build `SharedTensorParameter` from the Phase-5 `SharedSpecInput` variant in spec JSON, routing it through the suspended-compiler path without triggering a second clang parse. The lowering (SHLOWER-01/02) adds a per-operand branch in `ExternCallOpToLLVM.cpp` that passes shared operands directly as `ptr addrspace(3)` (bypassing the distributed `alloca+store+ptr` path) and applies accumulated subview offsets via `getShmemOffset` GEP. The frontend (SHAPI-01) relaxes `call_extern()`'s isinstance check and threads `memory_space: "shared"` through `arg_params`.

**Primary recommendation:** The codebase already has all the infrastructure — `SharedTensorParameter` bindings, `SharedSpecInput` in spec JSON, `getShmemAffineBase` in Utility.cpp, and `SharedLayout` Python types. Phase 6 is primarily about branching on `memory_space` at three integration points: Python spec-consumption loops, C++ lowering per-operand loop, and frontend isinstance guard.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-12:** At semantic time, `infer_result()` builds `SharedTensorParameter` with degenerate all-zero offset/block bases + default alignment — mirrors the degenerate-basis pattern for distributed args (`compiler.py:276-295`). Template deduction only needs T + Shape; real bases flow at llir stage from Phase-5 extracted specs.
- **D-13:** Phase 6 assumes shared-layout template parameter is DEDUCED (`template<class L> f(SharedTensor<T,S,L>&)`). No `PlaceholderSharedLayout` fallback.
- **D-14:** Inference hook distinguishes shared from distributed via `memory_space: "shared"` key in arg_params dict (absent/`"register"` for distributed).
- **D-15:** `TypeBuilder::BuildSharedTensor` applies clang address-space qualifier (`LangAS::cuda_shared`) to `SharedTensor&` pointee — no addrspacecast at call sites.
- **D-16:** Per-operand shared-vs-distributed branch is spec-JSON driven: `_pre_compile_extern_calls()` writes per-symbol per-arg memory-space list into `ttg.extern_call_arg_spaces` module attribute, parsed in lowering.
- **D-17:** Shared operands bypass `alloca+store+ptr` path: `getSharedMemoryObjectFromStruct` on promoted memdesc struct → base + `getShmemOffset` GEP → pass `ptr addrspace(3)` directly. Distributed operands keep existing path.
- **D-18:** `call_extern()` relaxes isinstance to accept `ttgl.shared_memory_descriptor`.
- **D-19:** Accepted shared layouts: `SharedLinearLayout`, `SwizzledSharedLayout`, `NVMMASharedLayout`; `PaddedSharedLayout` is rejected.
- **D-20:** Rejection error style matches f64 guard pattern.
- **D-21:** Shared-layout info reaches backend by passing the layout object via arg_params `layout` field.
- **D-22:** Lit-only compile-tier automation: one mixed-arg lit test for `ExternCallOpToLLVM`.
- **D-23:** Subview offset verification (SHLOWER-02) via manual LLVM IR dump inspection in Phase 6.
- **L-01 (landmine):** AS3 pointer erasure through store/reload — shared pointers must NOT route through stack slots. D-17 deliberately passes AS3 pointer directly.

### Agent's Discretion
- Exact module-attribute name for arg memory spaces (suggested: `ttg.extern_call_arg_spaces`)
- Whether shared branch uses `getShmemAffineBase` (one call) or `getSharedMemoryObjectFromStruct` + explicit `getShmemOffset` GEP (two steps)
- How `promoteOperands` interacts with memdesc operands
- Helper structure for SharedLayout→(offset_bases, block_bases, alignment) extraction

### Deferred Ideas (OUT OF SCOPE)
- AS3 pointer preservation across store/reload (MemorySSA-class analysis) — landmine L-01 only
- `PlaceholderSharedLayout` fallback
- Shared-memory return (SHRET-01), PaddedSharedLayout, dynamic `extern __shared__`, auto-barriers

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Frontend isinstance validation (SHAPI-01) | Python Frontend (`_semantic.py`) | — | Validation at user call site before IR build |
| Semantic-time inference for shared args (SHWIRE-01 semantic half) | Python Backend (`InferExternCallResult.infer_result`) | Clang AST (C++) | Degenerate-basis SharedTensorParameter → template deduction → return type |
| arg_params construction (SHAPI-01) | Python Frontend (`_semantic.py`) | — | Builds dict with memory_space key from descriptor type |
| Spec JSON consumption + SharedTensorParameter construction (SHWIRE-01 llir half) | Python Backend (`_pre_compile_extern_calls`) | — | Per-input branching on memory_space in spec JSON |
| Module-attr emission (SHLOWER-01 prep) | Python Backend (`_pre_compile_extern_calls`) | — | `ttg.extern_call_arg_spaces` StringAttr |
| Per-operand lowering branch (SHLOWER-01/02) | MLIR→LLVM C++ (`ExternCallOpToLLVM.cpp`) | — | Spec-JSON-driven: shared → direct AS3 ptr, distributed → alloca+store+ptr |
| Subview offset GEP (SHLOWER-02) | MLIR→LLVM C++ (`Utility.cpp`) | — | `getShmemOffset` / `getShmemAffineBase` |
| SharedTensor clang type with addrspace (D-15) | Clang AST C++ (`clang_compiler.cc`) | — | `LangAS::cuda_shared` on pointee |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Existing codebase (Phase 4-5) | HEAD | `SharedTensorParameter`, `SharedSpecInput`, `getShmemAffineBase` | Already integrated and tested |
| LLVM 23 (project-pinned) | 23.x | Address space support, `LangAS::cuda_shared` | Project's self-compiled LLVM |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pybind11 | (project) | `llvm.SharedTensorParameter` binding already exists | Python↔C++ shared param routing |
| LLVM JSON parser | (project) | Module attribute parsing in C++ lowering | `ttg.extern_call_arg_spaces` attribute read |

**Installation:** No new packages required. All infrastructure exists in the codebase from Phases 4-5.

## Package Legitimacy Audit

> Skipped — Phase 6 does not install external packages. All work modifies existing source files. No new `pip install` / `npm install` / `cargo add` commands.

## Architecture Patterns

### System Architecture Diagram

```
gl.call("tt_plugin.cu", "add_bias", tensor_a, shared_desc, result_layout=...)
  │
  │ _core.py:775-811 — gl.call builtin
  │   └─ args = [tensor_a.handle, shared_desc.handle]   // no to_tensor for shared
  ▼
_semantic.py:250 call_extern()
  │
  │ isinstance(a, (tensor, shared_memory_descriptor))  — SHAPI-01: relaxed guard
  │ f64 guard on all args
  │ Build arg_params: [{dtype, shape, layout}, {dtype, shape, layout, memory_space:"shared"}]
  │   └─ infer_hook.infer_result(libpath, func, arg_params) — D-14
  │
  ├──► InferExternCallResult.infer_result()  (SHWIRE-01 semantic half)
  │     │
  │     │ Per-arg: if memory_space=="shared" → SharedTensorParameter(all-zero bases)
  │     │           else → TensorParameter(degenerate bases)
  │     │ CudaFuncRequest.param_types = [variant<TensorParameter, SharedTensorParameter>]
  │     │ compiler.infer([req]) → FunctionResolver with SharedTensor& params
  │     │ EvaluateFunctionReturnType → return dtype+shape
  │     ▼
  │     Returns [(scalar_name, shape), ...] to _semantic.py
  │
  │ result_types built from inferred dtype+shape
  │ builder.create_extern_call(src_path, func, arg_handles, result_ir_types, ...)
  │   → ttg.extern_call op in MLIR
  ▼
ttg.extern_call %tensor, %memdesc : (tensor<...>, !ttg.memdesc<...>) -> tensor<...>
  │
  │ extractExternCallSpecs() — Phase 5: extracts both TensorSpecInput and SharedSpecInput
  │ SharedSpecInput has memory_space:"shared", offset_bases, block_bases, alignment
  ▼
_pre_compile_extern_calls()  (SHWIRE-01 llir half)
  │
  │ Per-input in spec JSON:
  │   if "memory_space"=="shared":
  │     stp = llvm.SharedTensorParameter()
  │     stp.type = _scalar_type_for(dtype); stp.shape = shape
  │     stp.offset_basis = spec["offset_bases"]; stp.block_basis = spec["block_bases"]
  │     stp.alignment = spec["alignment"]; stp.layout_rank = len(shape)
  │     → append to req.param_types
  │   else:
  │     tp = llvm.TensorParameter() — existing path
  │
  │ Write ttg.extern_call_arg_spaces module attribute: {"symbol":["register","shared","register"]}
  │
  │ Suspended compiler: _hook.compile_bitcode(libpath, requests)
  │   compileBitcode → BuildSharedTensor(LangAS::cuda_shared) → codegen
  │   → mangled callee takes ptr addrspace(3) natively
  │
  │ link_cuda_bitcode → CloneFunctionInto
  ▼
llir stage: make_llir() links bitcodes, O3 inlines, verifies
  │
  │ Parse count assertion: delta == distinct_cu (line 683) — single-parse guard
  ▼
convert-triton-gpu-to-llvm pass → ExternCallOpToLLVM.cpp
  │
  │ Parse ttg.extern_call_arg_spaces JSON per symbol (SHLOWER-01)
  │ promotedOperands = converter->promoteOperands(...)  (line 137)
  │
  │ FOR each operand i:
  │   IF arg_spaces[i] == "shared":
  │     // SHLOWER-01: bypass alloca+store+ptr path
  │     obj = getSharedMemoryObjectFromStruct(promotedOperands[i])
  │     // SHLOWER-02: apply subview offset
  │     promotedOperands[i] = getShmemAffineBase(obj, srcMemDescTy)
  │       → GEP base + shmemOffset → ptr addrspace(3)
  │   ELSE:
  │     // existing distributed path: alloca + store + ptr → ptr addrspace(0)
  │     alloca stack slot + store struct + use ptr
  │
  │ Build call: @mangled_fn(ptr addrspace(0), ptr addrspace(3), ptr addrspace(0))
  ▼
LLVM IR: call @_Z8add_bias... (ptr %a_alloca, ptr addrspace(3) %shmem_base, ptr %b_alloca)
```

### Recommended Project Structure

No new files. Modified files:

```
third_party/nvidia/backend/compiler.py        # SHWIRE-01: spec-consumption loops
python/triton/experimental/gluon/language/
├── _core.py                                   # SHAPI-01: gl.call() arg dispatching
├── _semantic.py                               # SHAPI-01: isinstance relaxation + arg_params
lib/Conversion/TritonGPUToLLVM/
└── ExternCallOpToLLVM.cpp                     # SHLOWER-01/02: per-operand shared branch
test/TritonGPU/
└── extern-call-shared-args.mlir               # D-22: new mixed-arg lowering lit test
```

### Pattern 1: Spec-JSON–Driven Branching (SHWIRE-01)

**What:** Python-side consumption of Phase-5 spec JSON where each input dict carries a `memory_space` key (`"shared"` or absent for register).

**When to use:** Both `_pre_compile_extern_calls()` spec-consumption loops — suspended path (line 786-796) and fallback path (line 820-832).

**Existing pattern to extend** [CITED: compiler.py:786-796]:
```python
# Current code builds only TensorParameter per input:
for inp in spec_entry["inputs"]:
    tp = llvm.TensorParameter()
    tp.type = _scalar_type_for(inp["dtype"])
    tp.shape = inp["shape"]
    # ... distributed bases ...
    param_types.append(tp)

# Phase 6 extension:
for inp in spec_entry["inputs"]:
    if inp.get("memory_space") == "shared":
        stp = llvm.SharedTensorParameter()
        stp.type = _scalar_type_for(inp["dtype"])
        stp.shape = inp["shape"]
        stp.offset_basis = inp.get("offset_bases", [])
        stp.block_basis = inp.get("block_bases", [])
        stp.alignment = inp.get("alignment", 16)
        stp.layout_rank = len(inp["shape"])
        param_types.append(stp)
    else:
        tp = llvm.TensorParameter()
        # ... existing path unchanged ...
        param_types.append(tp)
```

### Pattern 2: Module-Attribute JSON Side-Channel (SHLOWER-01)

**What:** Pass per-symbol per-arg memory spaces from Python backend to C++ lowering via a `StringAttr` JSON module attribute.

**Existing pattern** [CITED: ExternCallOpToLLVM.cpp:13-69]:
```cpp
// Existing pattern for reading mangled names from module attribute:
auto attr = module->getAttrOfType<StringAttr>("ttg.extern_call_mangled_names");
auto json = llvm::json::parse(attr.getValue());
auto *obj = json->getAsObject();
auto it = obj->find(symbol);
mangledName = it->second.getAsString()->str();
```

**Phase 6 addition — Python side** (`_pre_compile_extern_calls`):
```python
# After building per-symbol param_types, record memory spaces:
arg_spaces_map = {}
for spec_entry in specs_list:
    symbol = spec_entry["symbol"]
    spaces = []
    for inp in spec_entry["inputs"]:
        spaces.append("shared" if inp.get("memory_space") == "shared" else "register")
    arg_spaces_map[symbol] = spaces

# Store as module attribute (same StringAttr pattern as extern_call_mangled):
llvm.set_module_string_attr(mod, "ttg.extern_call_arg_spaces", json.dumps(arg_spaces_map))
```

**Phase 6 addition — C++ side** (new function in `ExternCallOpToLLVM.cpp`):
```cpp
static LogicalResult getArgMemorySpaces(ModuleOp module,
    const std::string &symbol, std::vector<std::string> &spaces) {
  // Same JSON-parsing pattern as getMangledName (lines 13-39)
  // Parse ttg.extern_call_arg_spaces attribute
  // Extract per-symbol array from JSON object
}
```

### Pattern 3: Per-Operand Shared vs. Distributed Branch (SHLOWER-01/02)

**What:** In the `matchAndRewrite` loop (currently lines 141-152, which applies `alloca+store+ptr` to ALL operands), add a per-operand branch keyed on the memory space from the module attribute.

**Existing code** [CITED: ExternCallOpToLLVM.cpp:141-152]:
```cpp
unsigned numTensorArgs = op.getInputs().size();
auto ptrTy = LLVM::LLVMPointerType::get(ctx, 0);
for (unsigned i = 0; i < numTensorArgs; ++i) {
    auto structVal = promotedOperands[i];
    auto structTy = structVal.getType();
    Value one = b.i32_val(1);
    auto *builder = &static_cast<OpBuilder &>(rewriter);
    Value stackPtr = LLVM::AllocaOp::create(
        *builder, loc, ptrTy, structTy, one, 0).getResult();
    b.store(structVal, stackPtr);
    promotedOperands[i] = stackPtr;
}
```

**Phase 6 modification — recommended approach** [CITED: Utility.cpp:1401-1408, 1428-1452]:
```cpp
// After promotedOperands, read arg spaces from module attribute:
std::vector<std::string> argSpaces;
(void)getArgMemorySpaces(module, op.getSymbol().str(), argSpaces);

auto ptrTyAS0 = LLVM::LLVMPointerType::get(ctx, 0);
for (unsigned i = 0; i < numTensorArgs; ++i) {
    bool isShared = (i < argSpaces.size() && argSpaces[i] == "shared");
    if (isShared) {
        // SHLOWER-01/02: extract base + offset GEP from promoted memdesc struct
        auto memDescType = cast<MemDescType>(op.getInputs()[i].getType());
        auto elemType = memDescType.getElementType();
        auto smemObj = getSharedMemoryObjectFromStruct(
            loc, promotedOperands[i], elemType, rewriter);
        // getShmemAffineBase: base + getShmemOffset GEP in one call
        promotedOperands[i] = smemObj.getShmemAffineBase(
            loc, rewriter, memDescType);
        // Result is ptr addrspace(3) — passes directly to callee
    } else {
        // Existing distributed path unchanged
        auto structVal = promotedOperands[i];
        auto structTy = structVal.getType();
        Value one = b.i32_val(1);
        auto *builder = &static_cast<OpBuilder &>(rewriter);
        Value stackPtr = LLVM::AllocaOp::create(
            *builder, loc, ptrTyAS0, structTy, one, 0).getResult();
        b.store(structVal, stackPtr);
        promotedOperands[i] = stackPtr;
    }
}
```

### Anti-Patterns to Avoid
- **Routing shared pointers through alloca/store/load:** Triggers L-01 AS3 erasure. D-17 deliberately passes AS3 ptr directly — no stack slot for shared operands.
- **Hardcoding memory space in C++ lowering:** Use spec-JSON via module attribute (D-16), not an assumption based on operand type. The lowering should be driven by what the Python backend recorded.
- **Changing `numTensorArgs`:** The loop bound is `op.getInputs().size()`, which correctly counts both tensor and memdesc operands after Phase-5 ODS relaxation.
- **Second clang parse:** Don't create a new compiler for a `.cu` that already has a suspended compiler. The suspended-compiler path check at line 776-806 must continue to match by libpath.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Shared memory base+offset computation | Custom GEP math | `getShmemAffineBase` (Utility.cpp:1401) | Handles pseudoinvert, subview offsets, all layout types |
| Memdesc struct extraction | Manual extract_value | `getSharedMemoryObjectFromStruct` (Utility.cpp:1428) | Correctly counts bases vs offsets from struct body |
| Module-attribute JSON parsing in C++ | Custom string parsing | Copy `getMangledName` pattern (ExternCallOpToLLVM.cpp:13-39) | Already handles error cases, null checks, key lookup |
| Shared layout→bases extraction in Python | Manual layout math | Access `a.type.layout` object (D-21); build helper analogous to `_scalar_type_for` | Layout objects already have all data; don't re-derive |

## Runtime State Inventory

> SKIPPED — Phase 6 is a greenfield feature addition (adding shared-memory support to existing `extern_call` pipeline). No renames, refactors, or migrations. Existing distributed-only paths are preserved unchanged.

## Common Pitfalls

### Pitfall 1: `__init__.py` Export for `shared_memory_descriptor`

**What goes wrong:** `_semantic.py` imports `_core as ttgl`, and `ttgl.shared_memory_descriptor` is used in isinstance check. If `shared_memory_descriptor` is not exported from `__init__.py`, it won't be accessible.

**Why it happens:** The isinstance check at `_semantic.py:253-255` currently uses `ttgl.tensor`; `shared_memory_descriptor` must be in the namespace imported by `_semantic.py`.

**How to avoid:** Verify: `shared_memory_descriptor` IS already exported at `__init__.py:32` [VERIFIED: grep]. No action needed — already present.

**Warning signs:** `NameError: name 'shared_memory_descriptor' is not defined` at import time.

### Pitfall 2: `to_tensor()` Called on `shared_memory_descriptor` in `gl.call()`

**What goes wrong:** `_core.py:803` calls `_semantic.to_tensor(a)` on every arg, which raises `TypeError` for `shared_memory_descriptor` (not a tensor, not a scalar).

**Why it happens:** `to_tensor` in `triton/language/semantic.py:118-127` only accepts `self.tensor` or scalar constants — a `shared_memory_descriptor` triggers the `check_type` branch and raises.

**How to avoid:** In `gl.call()` (`_core.py:803`), replace:
```python
tensors = [_semantic.to_tensor(a) for a in args]
```
with:
```python
tensors = []
for a in args:
    if isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor)):
        tensors.append(a)
    else:
        tensors.append(_semantic.to_tensor(a))
```
Then `call_extern` receives `shared_memory_descriptor` objects directly with their `.handle` attribute intact.

**Warning signs:** `TypeError: cannot convert shared_memory_descriptor<...> of type <class '...shared_memory_descriptor'> to tensor`

### Pitfall 3: `promoteOperands` Interaction with MemDesc Operands

**What goes wrong:** `promoteOperands` (TypeConverter) converts memdesc operands, but the converted struct may need special handling. Shared operands need the raw promoted struct for `getSharedMemoryObjectFromStruct`.

**Why it happens:** `promoteOperands` calls `LLVMTypeConverter::promoteOperands`, which applies type conversion. For memdesc types, `convertMemDescType` (TypeConverter.cpp:56-89) produces a struct `{ptr, ptr, ..., i32, i32, ...}` with base pointers first and offsets after. Using this struct directly with `getSharedMemoryObjectFromStruct` is correct.

**How to avoid:** Pass `promotedOperands[i]` directly to `getSharedMemoryObjectFromStruct` — the promoted struct of a memdesc type IS the `SharedMemoryObject` struct the utility expects. This is the canonical usage pattern at `Utility.cpp:674`.

**Warning signs:** `assert(numBases > 0)` failure in `getSharedMemoryObjectFromStruct` — indicates wrong struct type passed.

### Pitfall 4: TypeConverter Base Pointer Already Has Correct Address Space

**What goes wrong:** Concern that the base ptr extracted from promoted memdesc struct has the wrong address space.

**Why it happens:** `convertMemDescType` at TypeConverter.cpp:60-61 uses `targetInfo.getAddressSpace(type.getMemorySpace())` — for shared memory, NVIDIA targets return address space 3. So the base pointer in the promoted struct IS already `ptr addrspace(3)`. ✅ [VERIFIED: codebase]

**How to avoid:** No cast needed. The `getShmemAffineBase` GEP preserves the pointer's address space.

### Pitfall 5: Single-Parse Guard Assertion

**What goes wrong:** The assertion at `compiler.py:683` (`parse_count_delta == distinct_cu`) fails, indicating a double-parse.

**Why it happens:** The suspended-compiler path (lines 777-806) must match the SAME `libpath` keys that `create_and_suspend` used. If the spec JSON's `libpath` differs (e.g., relative vs absolute), a new compiler is created via the fallback path (line 808+), incrementing the parse counter.

**How to avoid:** Ensure `gl.call()` resolves paths consistently (it already does at `_core.py:796-799`: `Path(src_path).resolve()`). The pre-scan in `_runtime.py:82-85` also uses `.resolve()`. Both are already consistent. ✅

**Warning signs:** `AssertionError: extern CUDA parse count mismatch: 2 parse(s) for 1 distinct .cu file(s)`

## Code Examples

### SHAPI-01: Frontend isinstance Relaxation

```python
# _core.py:775-811 — gl.call() arg dispatching
# Current (line 803):
tensors = [_semantic.to_tensor(a) for a in args]

# Phase 6:
tensors = []
for a in args:
    if isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor)):
        tensors.append(a)
    else:
        tensors.append(_semantic.to_tensor(a))

# _semantic.py:253-255 — isinstance relaxation
# Current:
for a in args:
    _check(isinstance(a, ttgl.tensor), ...)

# Phase 6:
for a in args:
    _check(isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor)), ...)

# _semantic.py:270-276 — arg_params with memory_space key
# Phase 6 addition:
for a in args:
    if isinstance(a, ttgl.shared_memory_descriptor):
        shared_layout = a.type.layout
        # Extract bases from the Python layout object (agent's discretion for helper)
        offset_bases = _extract_offset_bases(shared_layout)
        block_bases = _extract_block_bases(shared_layout)
        alignment = _extract_alignment(shared_layout)
        arg_params.append({
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "layout": shared_layout,
            "memory_space": "shared",
            "offset_bases": offset_bases,
            "block_bases": block_bases,
            "alignment": alignment,
        })
    else:
        arg_params.append({
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "layout": a.type.layout,
        })
```

### SHLOWER-01/02: C++ Lowering Branch

```cpp
// ExternCallOpToLLVM.cpp — in matchAndRewrite, after promotedOperands (line 137)

// Read per-arg memory spaces from module attribute
std::vector<std::string> argSpaces;
if (failed(getArgMemorySpaces(module, op.getSymbol().str(), argSpaces))) {
    // If attr missing, all args are register (pre-Phase-6 or tensor-only)
    argSpaces.assign(numTensorArgs, "register");
}

auto ptrTyAS0 = LLVM::LLVMPointerType::get(ctx, 0);
for (unsigned i = 0; i < numTensorArgs; ++i) {
    bool isShared = (i < argSpaces.size() && argSpaces[i] == "shared");
    if (isShared) {
        // Extract shared memory object from promoted struct
        auto memDescType = cast<MemDescType>(op.getInputs()[i].getType());
        auto llvmElemTy = getTypeConverter()->convertType(memDescType.getElementType());
        auto smemObj = getSharedMemoryObjectFromStruct(
            loc, promotedOperands[i], llvmElemTy, rewriter);
        // Apply subview offset + get base GEP → ptr addrspace(3)
        promotedOperands[i] = smemObj.getShmemAffineBase(loc, rewriter, memDescType);
    } else {
        // Existing distributed path: alloca + store + ptr
        auto structVal = promotedOperands[i];
        auto structTy = structVal.getType();
        Value one = b.i32_val(1);
        auto *builder = &static_cast<OpBuilder &>(rewriter);
        Value stackPtr = LLVM::AllocaOp::create(
            *builder, loc, ptrTyAS0, structTy, one, 0).getResult();
        b.store(structVal, stackPtr);
        promotedOperands[i] = stackPtr;
    }
}
```

### SHWIRE-01: InferResult with SharedTensorParameter

```python
# compiler.py:270-295 — infer_result() degenerate-basis construction
# Phase 6 extension (in the param_types loop):
for ap in arg_params:
    if ap.get("memory_space") == "shared":
        stp = llvm.SharedTensorParameter()
        stp.type = _scalar_type_for(ap["dtype"])
        stp.shape = ap["shape"]
        # Degenerate all-zero bases for template deduction only (D-12)
        stp.offset_basis = []
        stp.block_basis = []
        stp.alignment = 16
        stp.layout_rank = len(ap["shape"])
        param_types.append(stp)
    else:
        tp = llvm.TensorParameter()
        # ... existing degenerate-basis construction unchanged ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Tensor-only `extern_call` | Mixed tensor+memdesc operands | Phase 5 (ODS relaxation) | Downstream passes see both types |
| All operands lowered via alloca+store+ptr | Shared operands bypass alloca path, pass AS3 ptr directly | Phase 6 (this phase) | Correct addrspace for shared memory |
| Layout inferred from first_input only | Return type inferred via CUDA template deduction | v1.0 (Phases 1-3) | Type-consistent downstream IR |

**Deprecated/outdated:**
- None — Phase 6 adds a branch to existing paths; no existing paths are removed.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | LLVM lit (FileCheck) + pytest |
| Config file | `test/lit.cfg.py`, `pyproject.toml` |
| Quick run command | `cd build && ninja triton-opt && lit -v test/TritonGPU/extern-call-shared-args.mlir` |
| Full suite command | `make test-lit` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SHWIRE-01 | `_pre_compile_extern_calls` builds SharedTensorParameter + single-parse guard holds | integration (Python) | Manual verification via parse-count assertion at runtime | ❌ Wave 0 (implicit via parse guard; no separate test needed) |
| SHLOWER-01 | `ptr addrspace(3)` emitted for shared args, `ptr addrspace(0)` for tensors | lit (MLIR→LLVM) | `lit -v test/TritonGPU/extern-call-shared-args.mlir` | ❌ Wave 0 |
| SHLOWER-02 | Subview offset GEP applied → callee gets `base + shmemOffset` | lit (MLIR→LLVM) + manual IR inspection | `lit -v test/TritonGPU/extern-call-shared-args.mlir` + manual dump | ❌ Wave 0 |
| SHAPI-01 | `gl.call()` accepts `shared_memory_descriptor` arg | unit (pytest) | Can verify via lit test that exercises the whole pipeline; no standalone unit test needed | ❌ Wave 0 (validated by full pipeline lit test) |

### Sampling Rate
- **Per task commit:** `cd build && ninja triton-opt && lit -v test/TritonGPU/extern-call-shared-args.mlir`
- **Per wave merge:** `make test-lit`
- **Phase gate:** All lit tests green + parse-count assertion verified

### Wave 0 Gaps
- [ ] `test/TritonGPU/extern-call-shared-args.mlir` — mixed shared+distributed args lowering test with FileCheck
- [ ] Framework install: `ninja triton-opt` — already built as part of Phase 5 development
- [ ] No separate Python unit tests needed — validation is via lit test + full pipeline

### Lit Test Design (D-22)

New file: `test/TritonGPU/extern-call-shared-args.mlir`

```mlir
// RUN: triton-opt %s -split-input-file -convert-triton-gpu-to-llvm | FileCheck %s
//
// SHLOWER-01/02: Mixed shared+distributed args in ttg.extern_call.
// Verifies: (1) distributed operands get alloca+store+ptr (AS0)
//           (2) shared operands get ptr addrspace(3) directly
//           (3) subview offset GEP is applied to shared base

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [1, 0]]}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {
  "ttg.target" = "cuda:0",
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 1 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.extern_call_mangled_names" = "{\"test_mixed\": \"_Z10test_mixed...\"}",
  "ttg.extern_call_arg_spaces" = "{\"test_mixed\": [\"register\", \"shared\"]}"
} {
  tt.func @mixed_shared_tensor_extern_call(
      %a: tensor<32x64xf32, #blocked>,
      %b: !ttg.memdesc<4x4xf32, #shared, #smem>) {
    // CHECK: alloca
    // CHECK: store
    // Expected: first arg (distributed) gets alloca+store
    // CHECK: ptr addrspace(3)
    // Expected: second arg (shared) is ptr addrspace(3)
    // CHECK: llvm.call @_Z10test_mixed
    %result = ttg.extern_call %a, %b
        : (tensor<32x64xf32, #blocked>, !ttg.memdesc<4x4xf32, #shared, #smem>)
        -> tensor<32x64xf32, #blocked>
        { symbol = "test_mixed", libpath = "test.cu" }
    tt.return
  }
}
```

## Security Domain

> Required (`security_enforcement` not explicitly `false` in config.json).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (compiler infrastructure, no user auth) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | isinstance guards in frontend; spec-JSON parsing in C++ with `dyn_cast`/`isa` checks; `llvm::json::parse` error handling |
| V6 Cryptography | no | — |

### Known Threat Patterns for Phase 6

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| `shared_memory_descriptor` handle misuse (wrong address space) | Tampering | D-17: direct AS3 ptr pass, no alloca; L-01: documented landmine |
| Invalid spec JSON deserialization | Tampering | `llvm::json::parse` returns `std::optional`; each access checked with `getAsObject`/`getAsString`/`getAsArray`; `dyn_cast` in `extractExternCallSpecs` (not `cast`) |
| Type confusion (shared vs distributed in lowering) | Tampering | D-16: per-operand branch driven by explicit `arg_spaces` module attribute, not by type inspection |
| Malformed module attribute JSON | Denial of Service | JSON parse failure → fallback to all-"register" in C++ (graceful degradation for pre-Phase-6 modules) |
| Frontend bypass (non-tensor args reaching IR builder) | Spoofing | isinstance guard at `_semantic.py:253-255` catches before IR construction |

## Sources

### Primary (VERIFIED — from codebase)
- `third_party/nvidia/backend/compiler.py:270-295, 676-684, 709-872` [VERIFIED: codebase grep] — `infer_result`, single-parse guard, `_pre_compile_extern_calls`
- `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp:1-293` [VERIFIED: codebase grep] — full lowering, module-attr JSON pattern, alloca+store+ptr loop
- `lib/Conversion/TritonGPUToLLVM/Utility.cpp:1366-1452` [VERIFIED: codebase grep] — `getShmemOffset`, `getShmemAffineBase`, `getSharedMemoryObjectFromStruct`
- `lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp:56-89` [VERIFIED: codebase grep] — `convertMemDescType` produces struct with AS3 base ptr
- `python/src/clang_compiler.cc:494-525, 1014-1032, 1216-1237, 1297-1317` [VERIFIED: codebase grep] — `BuildSharedTensor`, `CUDACompiler::BuildSharedTensor`, variant dispatch in `inferReturnTypes`/`compileBitcode`
- `python/src/clang_compiler.cc:1421-1623` [VERIFIED: codebase grep] — `extractExternCallSpecs` with TensorSpecInput/SharedSpecInput variant
- `python/src/clang_compiler.h:142-191` [VERIFIED: codebase grep] — `SharedLayoutInfo`, `SharedTensorParameter`, `CudaFuncRequest::ParamTypes` variant
- `python/src/llvm.cc:1007-1042` [VERIFIED: codebase grep] — `llvm.SharedTensorParameter` pybind binding
- `python/triton/experimental/gluon/language/_semantic.py:250-319` [VERIFIED: codebase grep] — `call_extern` isinstance check, arg_params, result types
- `python/triton/experimental/gluon/language/_core.py:185-219, 291-329, 775-811` [VERIFIED: codebase grep] — `shared_memory_descriptor_type`, `shared_memory_descriptor`, `gl.call`
- `python/triton/experimental/gluon/language/__init__.py:32` [VERIFIED: codebase grep] — `shared_memory_descriptor` is exported
- `python/triton/language/semantic.py:118-127` [VERIFIED: codebase grep] — `to_tensor` raises TypeError for non-tensor, non-scalar
- `python/triton/experimental/gluon/_runtime.py:67-99` [VERIFIED: codebase grep] — pre-scan for gl.call patterns + create_and_suspend
- `test/TritonGPU/extern-call-mixed-inputs.mlir:1-20` [VERIFIED: codebase grep] — Phase-5 lit test pattern for mixed operands
- `python/src/gluon_ir.cc:191-280, 615-624` [VERIFIED: codebase grep] — `layoutToGluon`, `create_extern_call`

### Secondary (CITED — inferred from codebase patterns)
- Module attribute naming convention: `ttg.extern_call_*` prefix used for `mangled_names`, `extractor_names` — extending to `ttg.extern_call_arg_spaces` follows convention
- `_serialize_return_types` helper at `compiler.py` — existing pattern for module-attr + JSON serialization

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `ttg.extern_call_arg_spaces` attribute name chosen at agent's discretion | Architecture Patterns | Low — rename in Python + C++ is trivial; no API surface |
| A2 | `getShmemAffineBase` works correctly for non-partitioned shared memory (asserts `bases.size() == 1`) | Architecture Patterns | Low — Phase 6 uses fixed-shape single-allocation shared memory; partitioned shared not in scope |
| A3 | `promoteOperands` correctly converts memdesc to struct format expected by `getSharedMemoryObjectFromStruct` | Common Pitfalls | Low — existing usage at `Utility.cpp:674` confirms this pattern works |
| A4 | `TypeBuilder::BuildSharedTensor` with LangAS::cuda_shared produces correct IR — verified existentially from Phase 4 implementation but not re-verified in this session | Architecture Patterns | Low — Phase 4 AST round-trip tests passed |
| A5 | Helper for extracting `offset_bases`/`block_bases`/`alignment` from Python `SharedLayout` objects can use `layout._to_ir()` / `layoutToGluon` reverse path or direct attribute access | Code Examples | Low — layout objects are frozen dataclasses with public attributes |

## Open Questions

1. **SharedLayout → bases extraction helper design**
   - What we know: `SharedLinearLayout`, `SwizzledSharedLayout`, `NVMMASharedLayout` all have `toLinearLayout` conversion (Phase 5 verified). Python layout objects are frozen dataclasses.
   - What's unclear: Whether the helper extracts from the Python object directly or rounds through MLIR→gluon→extract.
   - Recommendation: Extract directly from Python layout attributes (e.g., `layout.offset_bases` for `SharedLinearLayout`). The `SwizzledSharedLayout` and `NVMMASharedLayout` types can be converted via `toLinearLayout` → create a temporary `SharedLinearLayout` → extract. This is agent's discretion.

2. **`set_module_string_attr` existence**
   - What we know: `ir.cc` has `set_str_attr` on Operation/ModuleOp [CITED: AGENTS.md]. The mangled names are stored via the existing attribute pattern.
   - What's unclear: Does a dedicated `set_module_string_attr` helper exist, or does the C++ code use `mod->setAttr("key", StringAttr::get(ctx, json_str))` directly?
   - Recommendation: The lowering already reads `module->getAttrOfType<StringAttr>("ttg.extern_call_mangled_names")` — the Python side should write with the same pattern. Check `ir.cc` for a binding or use `mod.set_attr()`.

3. **SRet (struct return) path interaction with shared args**
   - What we know: `ExternCallOpToLLVM.cpp:171-234` handles tuple returns with sret. The `mainArgs` array (line 197-199) appends `promotedOperands` after the sret pointer.
   - What's unclear: Whether the shared branch in `promotedOperands` modification (replacing struct with AS3 ptr) interacts correctly with the sret path — the call signature builder uses `promotedTypes` from modified `promotedOperands`.
   - Recommendation: The shared branch replaces the memdesc struct Value with a `ptr addrspace(3)` Value in the same `promotedOperands` vector. Both the non-tuple (line 248) and sret (line 194) paths build function types from `promotedTypes` which is derived from `promotedOperands` after modification — this should be correct. Lit test should cover single-result (non-tuple) first; tuple+shared is a Phase 7 concern per D-22.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| triton-opt | lit tests | ✓ | build output | build via `ninja triton-opt` |
| clang++ (self-compiled LLVM) | clang_compiler.cc build | ✓ | 23.x (pinned) | — |
| lit (FileCheck) | lit test validation | ✓ | (llvm project) | — |
| Python 3 | compiler.py runtime | ✓ | (system) | — |

**Missing dependencies with no fallback:** None — all dependencies are part of the existing build.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all infrastructure from Phases 4-5 verified via codebase grep
- Architecture: HIGH — all three integration points (Python spec loop, C++ lowering loop, frontend isinstance) have clear existing patterns to extend; verified at file:line level
- Pitfalls: HIGH — AS3 erasure landmine (L-01) explicitly documented by user; `to_tensor` TypeError confirmed via codebase read; promoteOperands interaction confirmed via existing usage at Utility.cpp:674

**Research date:** 2026-07-16
**Valid until:** 2026-08-16 (30 days — stable compiler infrastructure)
