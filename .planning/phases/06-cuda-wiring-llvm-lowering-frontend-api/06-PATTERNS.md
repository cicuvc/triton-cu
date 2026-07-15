# Phase 06: CUDA Wiring + LLVM Lowering + Frontend API - Pattern Map

**Mapped:** 2026-07-16
**Files analyzed:** 7 (5 modified, 1 new)
**Analogs found:** 7 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `python/triton/experimental/gluon/language/_core.py` (modify) | frontend builtin | request-response | same file: `gl.call()` (lines 775-811) | exact |
| `python/triton/experimental/gluon/language/_semantic.py` (modify) | semantic builder | request-response | same file: `call_extern()` (lines 250-319) | exact |
| `third_party/nvidia/backend/compiler.py` (modify) | compiler backend | batch/transform | same file: `infer_result()` (270-295), `_pre_compile_extern_calls()` (786-796), `make_llir()` (625-630) | exact |
| `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` (modify) | MLIR→LLVM lowering | transform | same file: `getMangledName` (13-39), alloca+store loop (141-152) | exact |
| `python/src/clang_compiler.cc` & `.h` (modify) | clang AST utility | type construction | same file: `BuildTensor`, `TypeBuilder::BuildSharedTensor` (already exists) | exact |
| `test/TritonGPU/extern-call-shared-args.mlir` (new) | lit test | N/A | `test/TritonGPU/extern-call-mixed-inputs.mlir` | exact |

## Pattern Assignments

### 1. `python/triton/experimental/gluon/language/_core.py` — gl.call() arg dispatching (SHAPI-01)

**Analog:** `_core.py:775-811` — the existing `gl.call()` function

**Modification:** Replace the `to_tensor` call that raises `TypeError` on `shared_memory_descriptor`.

**Imports pattern** (lines 1-17):
```python
# No new imports needed — isinstance check uses ttgl.tensor and ttgl.shared_memory_descriptor,
# both already available via _core's own namespace.
```

**Core pattern — current arg processing** (line 803):
```python
    tensors = [_semantic.to_tensor(a) for a in args]
```

**Core pattern — Phase 6 replacement** (mirrors `_semantic.py:253-255` isinstance relaxation):
```python
    tensors = []
    for a in args:
        if isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor)):
            tensors.append(a)
        else:
            tensors.append(_semantic.to_tensor(a))
```

**Verification:** `shared_memory_descriptor` IS exported from `__init__.py` (line 32: `shared_memory_descriptor,`). The `ttgl` module alias in `_core.py` already includes it via `import triton.experimental.gluon.language as ttgl`.

---

### 2. `python/triton/experimental/gluon/language/_semantic.py` — call_extern() isinstance + arg_params (SHAPI-01)

**Analog:** `_semantic.py:250-319` — the existing `call_extern()` method

**Modifications:**
(a) Relax isinstance check (lines 253-255)
(b) Add `memory_space` key to arg_params (lines 270-276)
(c) Add PaddedSharedLayout rejection guard

**Pattern A — isinstance relaxation** (current lines 253-255):
```python
        for a in args:
            _check(isinstance(a, ttgl.tensor),
                   lambda: f"all arguments must be tensors but got {type(a)!r}")
```

**Pattern A — Phase 6 replacement:**
```python
        for a in args:
            _check(isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor)),
                   lambda: f"all arguments must be tensors or shared_memory_descriptors but got {type(a)!r}")
```

**Pattern B — PaddedSharedLayout rejection guard** (new, after f64 guard at lines 259-263):

**Analog for error style:** `_semantic.py:259-263` (f64 guard) + `compiler.py:754-758` (backend f64 guard)

Current f64 guard pattern (lines 259-263):
```python
        # Backend-agnostic f64 guard — raise before building IR.
        _f64_err = "gl.call() does not support float64; full Fp64 support is out of scope (see FP64-01)"
        for a in args:
            if hasattr(a, 'dtype') and str(a.dtype) in ("fp64", "f64", "float64"):
                raise NotImplementedError(_f64_err)
```

Phase 6 addition (after the f64 guard block):
```python
        # D-19/D-20: PaddedSharedLayout guard — raise before building IR.
        import triton.experimental.gluon.language._layouts as _glayouts
        _psl_err = ("gl.call() does not support PaddedSharedLayout shared memory; "
                     "use SharedLinearLayout/SwizzledSharedLayout/NVMMASharedLayout "
                     "(see v1.1 out-of-scope)")
        for a in args:
            if isinstance(a, ttgl.shared_memory_descriptor):
                if isinstance(a.type.layout, _glayouts.PaddedSharedLayout):
                    raise NotImplementedError(_psl_err)
```

**Pattern C — arg_params with memory_space key** (current lines 270-276):
```python
            arg_params = []
            for a in args:
                arg_params.append({
                    "dtype": str(a.dtype),
                    "shape": list(a.shape),
                    "layout": a.type.layout,
                })
```

Phase 6 replacement:
```python
            arg_params = []
            for a in args:
                if isinstance(a, ttgl.shared_memory_descriptor):
                    shared_layout = a.type.layout
                    # Extract bases from the Python layout object (D-21).
                    # All three accepted layout types can be normalized via toLinearLayout
                    # at the C++ side; Python side just passes the raw layout object
                    # plus the memory_space discriminator.
                    arg_params.append({
                        "dtype": str(a.dtype),
                        "shape": list(a.shape),
                        "layout": shared_layout,
                        "memory_space": "shared",
                    })
                else:
                    arg_params.append({
                        "dtype": str(a.dtype),
                        "shape": list(a.shape),
                        "layout": a.type.layout,
                    })
```

**Error handling pattern** — same as existing: `_check()` and `NotImplementedError` / `RuntimeError`, no `try/except` needed.

---

### 3. `third_party/nvidia/backend/compiler.py` — SHWIRE-01 spec-consumption + degenerate-basis + module attrs

**Analog:** `compiler.py:270-295` (degenerate-basis TensorParameter), `compiler.py:786-796` (suspended path spec loop), `compiler.py:820-832` (fallback path spec loop), `compiler.py:625-630` (module attr writing)

#### 3a. `InferExternCallResult.infer_result()` — degenerate SharedTensorParameter (D-12/D-13)

**Analog:** `compiler.py:270-295` — the existing degenerate TensorParameter construction

Current pattern (lines 270-295):
```python
        param_types = []
        for ap in arg_params:
            tp = llvm.TensorParameter()
            tp.type = _scalar_type_for(ap["dtype"])
            tp.shape = ap["shape"]
            tp.layout_shape = ap["shape"]
            # Compute minimally valid concrete bases for dtype+shape
            # inference. The exact layout doesn't matter — we just need a
            # valid Tensor<T,Shape,Layout<...>> type for template deduction.
            rank = len(ap["shape"])
            size = 1
            for d in ap["shape"]:
                size *= int(d)
            n_warps = 1
            _ffs_size_per_warp = (size // n_warps)
            _lsb_bit = (_ffs_size_per_warp & -_ffs_size_per_warp).bit_length() if _ffs_size_per_warp > 0 else 0
            n_lane_axes = 5
            n_reg_axes = max(0, _lsb_bit - n_lane_axes - 1)
            n_warp_axes = max(0, (n_warps & -n_warps).bit_length() - 1)
            tp.reg_basis = [0] * (n_reg_axes * rank)
            tp.lane_basis = [0] * (n_lane_axes * rank)
            tp.warp_basis = [0] * (n_warp_axes * rank)
            tp.n_warps = n_warps
            param_types.append(tp)
```

Phase 6 extension (inserted before `param_types.append`):
```python
        for ap in arg_params:
            if ap.get("memory_space") == "shared":
                # D-12: Degenerate all-zero SharedTensorParameter for
                # template deduction. Only T + Shape matter; real bases
                # flow at the llir stage from Phase-5 extracted specs.
                stp = llvm.SharedTensorParameter()
                stp.type = _scalar_type_for(ap["dtype"])
                stp.shape = ap["shape"]
                stp.offset_basis = []       # degenerate: all-zero bases
                stp.block_basis = []        # degenerate: all-zero bases
                stp.alignment = 16          # default alignment
                stp.layout_rank = len(ap["shape"])
                param_types.append(stp)
            else:
                tp = llvm.TensorParameter()
                # ... existing degenerate-basis construction unchanged ...
                param_types.append(tp)
```

**`llvm.SharedTensorParameter` API** (from `llvm.cc:1007-1042`):
```python
stp = llvm.SharedTensorParameter()
stp.type           # ScalarType enum
stp.shape          # list of uint32_t
stp.offset_basis   # list of uint32_t (flat)
stp.block_basis    # list of uint32_t (flat)
stp.layout_rank    # uint32_t
stp.alignment      # uint32_t
```

#### 3b. `_pre_compile_extern_calls()` — spec-consumption loops (SHWIRE-01 llir half)

**Analog:** `compiler.py:786-796` (suspended path) and `compiler.py:820-832` (fallback path)

**Suspended path** (lines 786-796) — current pattern:
```python
                    param_types = []
                    for inp in spec_entry["inputs"]:
                        tp = llvm.TensorParameter()
                        tp.type = _scalar_type_for(inp["dtype"])
                        tp.shape = inp["shape"]
                        tp.layout_shape = inp["shape"]
                        tp.reg_basis = inp.get("reg_bases", [])
                        tp.lane_basis = inp.get("lane_bases", [])
                        tp.warp_basis = inp.get("warp_bases", [])
                        tp.n_warps = inp.get("num_warps", 1)
                        param_types.append(tp)
```

**Suspended path — Phase 6 replacement:**
```python
                    param_types = []
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
                            tp.type = _scalar_type_for(inp["dtype"])
                            tp.shape = inp["shape"]
                            tp.layout_shape = inp["shape"]
                            tp.reg_basis = inp.get("reg_bases", [])
                            tp.lane_basis = inp.get("lane_bases", [])
                            tp.warp_basis = inp.get("warp_bases", [])
                            tp.n_warps = inp.get("num_warps", 1)
                            param_types.append(tp)
```

**Fallback path** (lines 820-832) — same modification: add `if inp.get("memory_space") == "shared"` branch with `SharedTensorParameter` construction before falling through to existing `TensorParameter` construction.

#### 3c. Module attribute emission for `ttg.extern_call_arg_spaces` (D-16)

**Analog:** `compiler.py:625-630` — existing module-attr JSON pattern for mangled names + extractor names

Current pattern (lines 625-630):
```python
        if has_extern_calls:
            mod.set_str_attr("ttg.extern_call_mangled_names",
                             _json.dumps(metadata["extern_call_mangled"]))
            if metadata.get("extern_call_extractor_names"):
                mod.set_str_attr("ttg.extern_call_extractor_names",
                                 _json.dumps(metadata["extern_call_extractor_names"]))
```

Phase 6 addition (inserted after extractor_names block, before `pm.run`):
```python
            # D-16: Per-symbol per-arg memory-space list for the C++ lowering.
            if metadata.get("extern_call_arg_spaces"):
                mod.set_str_attr("ttg.extern_call_arg_spaces",
                                 _json.dumps(metadata["extern_call_arg_spaces"]))
```

**Building `extern_call_arg_spaces` in `_pre_compile_extern_calls`** (after the spec-consumption loops, before storing module attrs):
```python
        # Build per-symbol per-arg memory-space lists for the C++ lowering.
        arg_spaces_map = {}
        for spec in specs_list:
            symbol = spec["symbol"]
            spaces = []
            for inp in spec["inputs"]:
                spaces.append("shared" if inp.get("memory_space") == "shared" else "register")
            arg_spaces_map[symbol] = spaces
        metadata["extern_call_arg_spaces"] = arg_spaces_map
```

**JSON shape:** `{"symbol_name": ["register", "shared", "register"], ...}`

---

### 4. `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` — SHLOWER-01/02 lowering

**Analog:** `ExternCallOpToLLVM.cpp:13-39` (JSON-parsing pattern), `ExternCallOpToLLVM.cpp:141-152` (alloca+store loop)

#### 4a. `getArgMemorySpaces` — new helper function (D-16)

**Analog:** `getMangledName` at lines 13-39 — JSON-parsing from module attribute

Current `getMangledName` pattern (lines 13-39):
```cpp
LogicalResult
getMangledName(ModuleOp module, const std::string &symbol,
               std::string &mangledName) {
  auto attr = module->getAttrOfType<StringAttr>(
      "ttg.extern_call_mangled_names");
  if (!attr)
    return failure();

  auto json = llvm::json::parse(attr.getValue());
  if (!json)
    return failure();

  auto *obj = json->getAsObject();
  if (!obj)
    return failure();

  auto it = obj->find(symbol);
  if (it == obj->end())
    return failure();

  auto val = it->second.getAsString();
  if (!val)
    return failure();

  mangledName = val->str();
  return success();
}
```

Phase 6 new function (copy structure, replace string result with vector):
```cpp
static LogicalResult getArgMemorySpaces(ModuleOp module,
    const std::string &symbol, std::vector<std::string> &spaces) {
  auto attr = module->getAttrOfType<StringAttr>(
      "ttg.extern_call_arg_spaces");
  if (!attr)
    return success();  // no spaces attr = all-register (pre-Phase-6 or tensor-only)

  auto json = llvm::json::parse(attr.getValue());
  if (!json)
    return success();

  auto *obj = json->getAsObject();
  if (!obj)
    return success();

  auto it = obj->find(symbol);
  if (it == obj->end())
    return success();

  auto arr = it->second.getAsArray();
  if (!arr)
    return success();

  for (auto &v : *arr) {
    auto s = v.getAsString();
    spaces.push_back(s ? s->str() : "register");
  }
  return success();
}
```

**Error handling pattern:** Returns `success()` on missing attr (graceful degradation for pre-Phase-6 modules). Only returns `failure()` on genuine parse errors (this function is lenient — caller handles missing data).

#### 4b. Per-operand shared branch in `matchAndRewrite` (SHLOWER-01/02)

**Analog:** `ExternCallOpToLLVM.cpp:141-152` — existing alloca+store+ptr loop

Current alloca+store+ptr loop (lines 141-152):
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

**Phase 6 replacement** (D-16/D-17 — insert after `promotedOperands`, before alloca loop):
```cpp
    unsigned numTensorArgs = op.getInputs().size();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 0);

    // D-16: Read per-arg memory spaces from module attribute.
    std::vector<std::string> argSpaces;
    (void)getArgMemorySpaces(module, op.getSymbol().str(), argSpaces);
    if (argSpaces.empty()) {
      // Fallback for pre-Phase-6 modules: all operands are register.
      argSpaces.assign(numTensorArgs, "register");
    }

    for (unsigned i = 0; i < numTensorArgs; ++i) {
      bool isShared = (i < argSpaces.size() && argSpaces[i] == "shared");
      if (isShared) {
        // SHLOWER-01/02: bypass alloca+store+ptr path.
        // Extract shared memory object from promoted memdesc struct,
        // apply subview offset via getShmemAffineBase, pass AS3 ptr
        // directly to callee (no stack slot — avoids L-01 AS3 erasure).
        auto memDescType = cast<MemDescType>(op.getInputs()[i].getType());
        auto llvmElemTy = getTypeConverter()->convertType(
            memDescType.getElementType());
        auto smemObj = getSharedMemoryObjectFromStruct(
            loc, promotedOperands[i], llvmElemTy, rewriter);
        promotedOperands[i] = smemObj.getShmemAffineBase(
            loc, rewriter, memDescType);
        // Result is ptr addrspace(3) — passes directly to callee.
      } else {
        // Existing distributed path: alloca + store + ptr (AS0).
        auto structVal = promotedOperands[i];
        auto structTy = structVal.getType();
        Value one = b.i32_val(1);
        auto *builder = &static_cast<OpBuilder &>(rewriter);
        Value stackPtr = LLVM::AllocaOp::create(
            *builder, loc, ptrTy, structTy, one, 0).getResult();
        b.store(structVal, stackPtr);
        promotedOperands[i] = stackPtr;
      }
    }
```

**Key utility functions used** (from `Utility.cpp`):

`getSharedMemoryObjectFromStruct` (lines 1428-1452):
```cpp
// Signature:
SharedMemoryObject getSharedMemoryObjectFromStruct(
    Location loc, Value llvmStruct, Type elemTy, RewriterBase &rewriter);

// Extracts bases (ptr) and offsets (i32) from the promoted memdesc struct.
// The struct layout from convertMemDescType (TypeConverter.cpp:56-89) is:
//   { ptr addrspace(3), ..., i32, i32, ... }
// where base pointer(s) come first, offsets follow.
```

`getShmemAffineBase` (lines 1401-1409):
```cpp
// Signature:
Value SharedMemoryObject::getShmemAffineBase(
    Location loc, RewriterBase &rewriter,
    triton::gpu::MemDescType srcTy) const;

// Implementation:
//   assert(bases.size() == 1);  // single-base (non-partitioned) shared memory
//   Value offset = getShmemOffset(loc, rewriter, srcTy);
//   return b.gep(bases[0].getType(), baseElemType, bases[0], offset);
// Produces: ptr addrspace(3) = base + shmemOffset GEP
```

**TypeConverter address space guarantee** (`TypeConverter.cpp:56-61`):
```cpp
Type TritonGPUToLLVMTypeConverter::convertMemDescType(
    MemDescType type, const TargetInfoBase &targetInfo) {
  auto ctx = type.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(
      ctx, targetInfo.getAddressSpace(type.getMemorySpace()));
  // For shared memory, NVIDIA targets return address space 3.
  // So the base pointer in the promoted struct IS already ptr addrspace(3).
```

**Imports required** (already present at lines 1-6):
```cpp
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/JSON.h"
```

---

### 5. `python/src/clang_compiler.cc` & `clang_compiler.h` — D-15 LangAS::cuda_shared

**Analog:** `clang_compiler.h:289-292` — `TypeBuilder::BuildSharedTensor` (already declared)

**The method `BuildSharedTensor` already exists** at `clang_compiler.h:290-292`:
```cpp
  clang::QualType BuildSharedTensor(clang::QualType ElementType,
                                    clang::QualType ShapeType,
                                    clang::QualType LayoutType);
```

**Implementation is in `clang_compiler.cc`** — the Phase 6 change is to apply a `LangAS::cuda_shared` address-space qualifier to the pointee type within the `BuildSharedTensor` implementation:

**Pattern in `BuildSharedTensor` implementation** (conceptual — verify actual implementation):
```cpp
clang::QualType CUDACompiler::BuildSharedTensor(const SharedTensorParameter &p) {
  // ... existing: build ElementType, ShapeType, LayoutType ...
  clang::QualType sharedTensorType = Builder.BuildSharedTensor(
      elemType, shapeType, layoutType);

  // D-15: Apply clang address-space qualifier to the SharedTensor& pointee
  // so the mangled callee signature natively takes ptr addrspace(3).
  clang::QualType as3Type = Ctx.getAddrSpaceQualType(
      sharedTensorType, clang::LangAS::cuda_shared);
  return Ctx.getLValueReferenceType(as3Type);
}
```

**Note:** AGENTS.md mentions this was already integrated in Phase 4. Verify the current implementation matches; if LangAS::cuda_shared is already applied, no change needed here.

---

### 6. `test/TritonGPU/extern-call-shared-args.mlir` — new lit test (D-22)

**Analog:** `test/TritonGPU/extern-call-mixed-inputs.mlir` (Phase 5) and `test/TritonGPU/extern-call-tensor-only.mlir`

**Test structure pattern** (from `extern-call-mixed-inputs.mlir:1-20`):
```mlir
// RUN: triton-opt %s -split-input-file -verify-diagnostics
//
// SHMLIR-01 positive test: Verifies that the ttg.extern_call ODS type constraint
// ... accepts mixed tensor+memdesc operands ...

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0], [2, 2]]}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_tensor_memdesc_extern_call(%a: tensor<...>, %b: !ttg.memdesc<...>) {
    // CHECK: ttg.extern_call
    %result = ttg.extern_call %a, %b : (...) -> ... { symbol = "test_fn", libpath = "test.cu" }
    tt.return
  }
}
```

**Phase 6 new file — `test/TritonGPU/extern-call-shared-args.mlir`:**
```mlir
// RUN: triton-opt %s -split-input-file -convert-triton-gpu-to-llvm | FileCheck %s
//
// SHLOWER-01/02: Mixed shared+distributed args in ttg.extern_call lowering.
// Verifies: (1) distributed operands get alloca+store+ptr (AS0)
//           (2) shared operands get ptr addrspace(3) directly
//           (3) subview offset GEP is applied to shared base
//           (4) both arg types coexist in same call preserving signature order

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [1, 0]]}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {
  "ttg.target" = "cuda:0",
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 1 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.extern_call_mangled_names" = "{\"test_mixed\": \"_Z10test_mixedPfP3U7cuda_sharedfi\"}",
  "ttg.extern_call_arg_spaces" = "{\"test_mixed\": [\"register\", \"shared\"]}"
} {
  tt.func @mixed_shared_tensor_extern_call(
      %a: tensor<32x64xf32, #blocked>,
      %b: !ttg.memdesc<4x4xf32, #shared, #smem>) {
    // First arg (distributed): expect alloca + store
    // CHECK: llvm.alloca
    // CHECK: llvm.store

    // Second arg (shared): should NOT have alloca; should result in ptr addrspace(3)
    // CHECK-NOT: llvm.alloca{{.*}}%{{.*}}memdesc
    // CHECK: ptr addrspace(3)

    // Final call must reference the correct mangled name
    // CHECK: llvm.call @_Z10test_mixed

    %result = ttg.extern_call %a, %b
        : (tensor<32x64xf32, #blocked>, !ttg.memdesc<4x4xf32, #shared, #smem>)
        -> tensor<32x64xf32, #blocked>
        { symbol = "test_mixed", libpath = "test.cu" }
    tt.return
  }
}
```

**Test execution:**
```bash
cd build && ninja triton-opt && lit -v test/TritonGPU/extern-call-shared-args.mlir
```

**lit.cfg.py** (already configured — no changes needed):
- Uses `-split-input-file` and `-convert-triton-gpu-to-llvm` pass
- FileCheck is available via lit

---

## Shared Patterns

### Authentication/Authorization
**Not applicable** — Phase 6 is a compiler pass + frontend, no user authentication.

### Error Handling

**Source — Python frontend:** `_semantic.py:253-255` (existing isinstance check pattern)
```python
_check(isinstance(a, ...), lambda: f"all arguments must be ...")
```

**Source — Python backend:** `compiler.py:754-758` (f64 guard + `ValueError`/`NotImplementedError` pattern)
```python
if dtype_str in ("f64", ...):
    raise NotImplementedError(
        "gl.call() does not support ...; ... (see FP64-01)")
raise ValueError(f"Unsupported dtype: {dtype_str}")
```

**Source — C++ lowering:** `ExternCallOpToLLVM.cpp:13-39` (JSON parse fallthrough pattern)
```cpp
if (!attr) return failure();  // or success() for optional attrs
if (!json) return failure();
```

**Apply to:** All files. Frontend uses `NotImplementedError` / `_check()`; backend uses `RuntimeError`; C++ uses `return failure()` / `op.emitError()`.

### Module Attribute Pattern

**Source:** `ExternCallOpToLLVM.cpp:13-39` + `compiler.py:625-630`

Python side (write):
```python
mod.set_str_attr("ttg.extern_call_<name>", json.dumps(data_dict))
```

C++ side (read):
```cpp
auto attr = module->getAttrOfType<StringAttr>("ttg.extern_call_<name>");
if (!attr) return failure();
auto json = llvm::json::parse(attr.getValue());
auto *obj = json->getAsObject();
auto it = obj->find(symbol);
```

**Apply to:** `compiler.py` (write `ttg.extern_call_arg_spaces`), `ExternCallOpToLLVM.cpp` (read `ttg.extern_call_arg_spaces`)

### Validation Pattern

**Source — Frontend isinstance guard:** `_semantic.py:253-255`
```python
_check(isinstance(a, ttgl.tensor), ...)
```

**Source — Backend guard pattern:** `compiler.py:754-758`
```python
if condition:
    raise NotImplementedError("descriptive message (see REQ-ID)")
```

**Apply to:** `_semantic.py` (relaxed isinstance + PaddedSharedLayout guard), `compiler.py` (`memory_space` branching)

### JSON Serialization Pattern

**Source:** `_serialize_return_types` at `compiler.py:164-182`
```python
def _serialize_return_types(return_type_map):
    scalar_names = { ... }
    result = {}
    for symbol, tp_list in return_type_map.items():
        result[symbol] = [{...} for tp in tp_list]
    return result
```

**Apply to:** Building `extern_call_arg_spaces` dict in `_pre_compile_extern_calls`.

---

## No Analog Needed

All files have exact analogs in the same files they modify — Phase 6 extends existing patterns rather than introducing new ones. No external analog is needed.

---

## Metadata

**Analog search scope:** `python/triton/experimental/gluon/language/`, `third_party/nvidia/backend/`, `lib/Conversion/TritonGPUToLLVM/`, `python/src/`, `test/TritonGPU/`
**Files scanned:** ~25 source files
**Pattern extraction date:** 2026-07-16
