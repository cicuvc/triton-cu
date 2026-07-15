# Phase 5: MLIR Op Relaxation + Spec Extraction - Research

**Researched:** 2026-07-15
**Domain:** MLIR ODS tablegen + C++ spec extraction (shared memory layout JSON)
**Confidence:** HIGH

## Summary

This phase makes two surgical changes: (1) relax the `ttg.extern_call` ODS operand constraint from `Variadic<TT_Tensor>` to `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>` at `TritonGPUOps.td:803`, and (2) replace the `cast<RankedTensorType>` at `clang_compiler.cc:1456` with a `dyn_cast` that branches on `MemDescType` to emit shared-layout JSON. The shared-layout path reuses the existing `toLinearLayout` infrastructure (fully wired for `SharedLinearEncodingAttr`, `SwizzledSharedEncodingAttr`, and `NVMMASharedEncodingAttr` at `LinearLayoutConversions.cpp:1323-1370`) to extract `offset_bases`, `block_bases`, and `alignment` — the same keys consumed by `gluon_ir.cc:238-242`.

**Primary recommendation:** The single-input-list ODS relaxation with a `std::variant<TensorSpecInput, SharedSpecInput>` per-input data model is the simplest and most MLIR-idiomatic approach. The shared extraction branch is a straightforward mirror of the distributed-layout branch with different dim-name constants (`"offset"`/`"block"` instead of `"register"`/`"lane"`/`"warp"`) and alignment from `SharedEncodingTrait::getAlignment()`.

## Project Constraints (from AGENTS.md)

- **Build:** `PYTHONPATH`-based dev (not `pip install -e .`), self-compiled LLVM at `/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/llvm-data/install`, `CC=clang CXX=clang++`, optional `lld` linker.
- **Canonical build:** `bash build.sh` (cmake + ninja). Build output: `build/libtriton.so` → copy to `python/triton/_C/libtriton.so`.
- **Lit tests:** `cd $BUILD_DIR && ninja triton-opt && lit -v test/<path>.mlir` (needs `triton-opt` rebuild after tablegen changes).
- **No `pip install -e .`** — overwrites venv's standard triton.
- **`clang_compiler.cc` compiled with `-fno-rtti`** (Clang libs are built without RTTI).
- **CMakeLists.txt:** Clang libs (CodeGen, Frontend, Driver, Sema, AST) and LLVMMIRParser are permanently in CMakeLists.txt.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| ODS operand type constraint relaxation | MLIR TableGen | — | ODS `.td` files define op signatures; generated C++ enforces types at parse time |
| Op verification (mixed tensor+memdesc) | MLIR Verifier | — | Auto-verified by `SameVariadicOperandSize` + type constraint from ODS |
| Spec extraction (`extractExternCallSpecs`) | C++ (clang_compiler.cc) | — | Scans MLIR module for `ttg.extern_call`, extracts operand types into C++ structs |
| Shared layout linearization (`toLinearLayout`) | C++ (LinearLayoutConversions.cpp) | — | Converts MLIR encoding attrs to LinearLayout; already supports shared encodings |
| JSON serialization (`tritonExtractExternCallSpecs`) | C++ (clang_compiler.cc) | — | `llvm::raw_string_ostream` manual JSON building |
| JSON consumption (`_pre_compile_extern_calls`) | Python (compiler.py) | — | Phase 6 — out of scope for Phase 5 but must tolerate unknown input keys |

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-09:** Use `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>` — single mixed input list with per-operand type branching.
- **D-10:** Replace single `SpecInput` struct with `std::variant<TensorSpecInput, SharedSpecInput>`. `TensorSpecInput`: `dtype`, `shape`, `numWarps`, `regBases`, `laneBases`, `warpBases`. `SharedSpecInput`: `dtype`, `shape`, `memory_space` ("shared"), `offset_bases`, `block_bases`, `alignment`. `ExternCallSpec::inputs` becomes `SmallVector<std::variant<TensorSpecInput, SharedSpecInput>, 4>`.
- **D-11:** Two lit tests: mixed tensor+memdesc parse, tensor-only regression parse. Extraction smoke testing deferred to Phase 6.

### Locked Upstream (from Phase 4 — do not re-decide)
- `SharedLinearLayout` is distinct from distributed `Layout`; carries `offset_bases` + `block_bases` + `alignment` (D-05)
- `SharedTensor` is argument-only for v1.1 — no shared memory return (D-03 scope)
- No new dialect op — relax the existing `ttg.extern_call`
- `toLinearLayout` already handles shared encodings at `gluon_ir.cc:1069`

### the agent's Discretion
- None specified — all key decisions are locked.
- JSON serialization mechanics within `tritonExtractExternCallSpecs()` (how `std::visit` dispatches the variant) are left to the planner.

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SHMLIR-01 | `ttg.extern_call` accepts `MemDescType` operands without breaking tensor-only path; lit test verifies | § ODS Relaxation, § Test Strategy |
| SHMLIR-02 | `extractExternCallSpecs()` handles `MemDescType` operands (no crash) and emits shared-layout specs | § Spec Extraction, § Shared Layout Extraction |

---

# Research Findings

## Current State

### 1. ODS Op Definition (`ttg.extern_call`)

**File:** `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:786-814`

```tablegen
def TTG_ExternCallOp : TTG_Op<"extern_call", [
    SameVariadicOperandSize
  ]> {
  let arguments = (ins
    Variadic<TT_Tensor>:$inputs,       // line 803 — must relax
    StrAttr:$symbol,
    StrAttr:$libpath,
    UnitAttr:$assert_no_conv,
    UnitAttr:$use_fast_math
  );
  let results = (outs Variadic<TT_Tensor>:$results);
  let assemblyFormat = [{
    $inputs `:` functional-type($inputs, $results) attr-dict
  }];
}
```

**Available includes** (already in this `.td` file):
- `TritonGPUTypes.td` (line 6) — defines `TTG_MemDescType`
- `TritonTypes.td` (line 11) — defines `TT_Tensor` (`RankedTensorOf<[TT_Float, TT_Int, TT_Ptr]>`)
- `mlir/IR/OpBase.td` (line 14) — provides `AnyTypeOf`, `SameVariadicOperandSize`

**Change needed:** Line 803: `Variadic<TT_Tensor>:$inputs` → `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs`

**Referenced patterns** (verified in codebase):
- `include/triton/Dialect/Triton/IR/TritonOps.td:877`: `Variadic<AnyTypeOf<[TT_Type]>>:$args` — same pattern, in this project
- `third_party/proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUOps.td:131`: `AnyTypeOf<[TTG_MemDescType, TT_Ptr]>:$buffer` — MemDescType in AnyTypeOf
- `third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td:56`: `Variadic<TTG_MemDescType>:$buffers` — Variadic with MemDescType

**Assembly format impact:** The `functional-type($inputs, $results)` already accepts mixed types — it simply prints `($inputs) -> ($results)`. No format change needed.

**Verifier impact:** The `TTG_Op` base class (line 27-29) automatically appends `VerifyTensorLayoutsTrait` and `VerifyMemDescLayoutsTrait` to all ops. The `extern_call` op already inherits both traits. With `SameVariadicOperandSize`, the verifier checks operand count consistency; with `AnyTypeOf`, it checks that each operand is either `TT_Tensor` or `TTG_MemDescType`. No downstream pass that does `isa<...>(operand.getType()).dyn_cast<RankedTensorType>(...)` should trigger the ODS type constraint — the ODS enforcement happens at parse time, and downstream passes get valid types.

### 2. Spec Extraction (`extractExternCallSpecs`)

**File:** `python/src/clang_compiler.cc:1417-1505`

**Current structs** (anonymous namespace, lines 1419-1435):
```cpp
struct SpecInput {                          // line 1421
  std::string dtype;
  llvm::SmallVector<int64_t, 4> shape;
  int64_t numWarps;
  llvm::SmallVector<int32_t, 16> regBases;  // line 1425
  llvm::SmallVector<int32_t, 16> laneBases;
  llvm::SmallVector<int32_t, 16> warpBases;
};

struct ExternCallSpec {                     // line 1430
  std::string symbol;
  std::string libpath;
  bool useFastMath = false;
  llvm::SmallVector<SpecInput, 4> inputs;   // line 1434 — becomes variant
};
```

**`extractExternCallSpecs(module)` (lines 1438-1505):** Walks all `ExternCallOp` in the module. For each operand (line 1455-1500):
1. **Line 1456:** `auto tensorTy = cast<RankedTensorType>(operand.getType());` — **THE CRASH**. MemDescType cannot be cast to RankedTensorType.
2. **Line 1457:** Gets `shape = tensorTy.getShape()`.
3. **Line 1458:** Gets `encoding = tensorTy.getEncoding()`.
4. **Line 1461:** Calls `toLinearLayout(shape, encoding)`.
5. **Lines 1463-1479:** Fills `SpecInput` with `shape`, `numWarps`, `regBases`, `laneBases`, `warpBases` using dim names `"register"`, `"lane"`, `"warp"` (constructed at line 1442-1447).
6. **Lines 1481-1497:** Maps `elementType` to dtype string (`"f32"`, `"f16"`, etc.).

**`tritonExtractExternCallSpecs(module)` (lines 1514-1567):** Manual JSON serialization via `llvm::raw_string_ostream` (line 1518). Emits per-input: `dtype`, `shape`, `num_warps`, `reg_bases`, `lane_bases`, `warp_bases` (lines 1536-1559).

**Includes available in clang_compiler.cc:**
- `triton/Dialect/TritonGPU/IR/Dialect.h` (line 35) — brings `MemDescType`, `SharedLinearEncodingAttr`, `SharedEncodingTrait`
- `triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h` (line 36) — brings `toLinearLayout(MemDescType)` overloads
- `triton/Tools/LinearLayout.h` (line 38) — `LinearLayout::getBases()`, `getInDimSize()`, `getInDimNames()`
- `clang_compiler.h` includes `<variant>` (line 36) — `std::variant` available

### 3. MemDescType API

**ODS definition:** `include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:23-84`

| Public method | Signature | Source |
|--------------|-----------|--------|
| `getShape()` | `ArrayRef<int64_t>` | ODS parameter `shape` (line 34) |
| `getElementType()` | `Type` | ODS parameter `elementType` (line 35) |
| `getEncoding()` | `Attribute` | ODS parameter `encoding` (line 36) |
| `getMemorySpace()` | `Attribute` | ODS parameter `memorySpace` (line 37) |
| `getMutableMemory()` | `bool` | ODS parameter `mutableMemory` (line 38) |
| `getAllocShape()` | `ArrayRef<int64_t>` | ODS parameter `allocShape` (line 39) |
| `getRank()` | `int64_t` (implied via `ShapedTypeInterface`) | `getShape().size()` |

### 4. Python Consumer (`_pre_compile_extern_calls`)

**File:** `third_party/nvidia/backend/compiler.py:709-867`

The consumer currently accesses (line 786-795):
```python
for inp in spec_entry["inputs"]:
    tp = llvm.TensorParameter()
    tp.type = _scalar_type_for(inp["dtype"])
    tp.shape = inp["shape"]
    tp.reg_basis = inp.get("reg_bases", [])
    tp.lane_basis = inp.get("lane_bases", [])
    tp.warp_basis = inp.get("warp_bases", [])
    tp.n_warps = inp.get("num_warps", 1)
```

**Impact:** Phase 5 does NOT modify this consumer (deferred to Phase 6). The new JSON keys (`memory_space`, `offset_bases`, `block_bases`, `alignment`) must not cause the existing consumer to crash — since it uses `.get()` with defaults, unknown keys are silently ignored. The variant JSON must include a discriminator (e.g., `"memory_space"` absent → tensor path; `"memory_space": "shared"` → shared path) so Phase 6 can distinguish.

### 5. Phase 4 Artifacts

**Phase 4 built:** Device-side C++ templates (`SharedLinearLayout`, `SharedTensor`), clang AST bridge (`TypeBuilder::BuildSharedLinearLayout/BuildSharedTensor`, `TypeInspector::DispatchTypeParsing` for shared, `FunctionResolver` with `SharedTensor&` params), `llvm.SharedTensorParameter` pybind11 binding.

Relevant to Phase 5:
- `clang_compiler.h:188` already has `SharedTensorParameter` in `CudaFuncRequest::ParamTypes` variant
- `clang_compiler.h:36` includes `<variant>`
- `llvm.cc` exposes `SharedTensorParameter` binding (verified Phase 4)

## Shared Layout Extraction

### How `toLinearLayout` Works for Shared Encodings

**File:** `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:1323-1370`

The primary overload `toLinearLayout(shape, layout)` branches on layout type:
```cpp
if (auto distributed = dyn_cast<DistributedEncodingTrait>(layout)) {
    result = distributed.toLinearLayout(shape);  // → dims: register, lane, warp
} else {
    // shared encodings:
    if (auto shared = dyn_cast<SwizzledSharedEncodingAttr>(layout))
        result = swizzledSharedToLinearLayout(shape, shared);           // line 1342
    else if (auto shared = dyn_cast<SharedLinearEncodingAttr>(layout))
        result = shared.toLinearLayout(shape);                         // line 1344
    else if (auto shared = dyn_cast<NVMMASharedEncodingAttr>(layout))
        result = nvmmaSharedToLinearLayout(shape, shared, ...);       // line 1347
    // ... more shared variants
}
```

**For MemDescType** (lines 1376-1392): `toLinearLayout(MemDescType)` extracts the allocShape (or shape), then calls the same `toLinearLayout(shape, encoding)` overload. All shared encodings produce a `LinearLayout` with **input dims `"offset"` and `"block"`**, not `"register"`/`"lane"`/`"warp"`.

### Dim Names for Shared Layouts

Confirmed by `python/src/gluon_ir.cc:238-242`:
```cpp
auto kOffset = mlir::StringAttr::get(ctx, "offset");
auto kBlock = mlir::StringAttr::get(ctx, "block");
return layouts.SharedLinearLayout(
    toStdVector(ll.getBases().lookup(kOffset)),
    toStdVector(ll.getBases().lookup(kBlock)),
    sharedLl.getAlignment());
```

And by `lib/Dialect/TritonGPU/IR/Dialect.cpp:3947-3948`:
```cpp
StringAttr kOffset = StringAttr::get(ctx, "offset");
StringAttr kBlock = StringAttr::get(ctx, "block");
```

**Key difference from distributed layouts:** Shared encodings use `"offset"` and `"block"` as input dims, NOT `"register"`/`"lane"`/`"warp"`. There is no `numWarps` concept — the `toLinearLayout` for shared layouts does not have a `kWarp` input dim.

### Extracting offset_bases and block_bases

For a memdesc operand with shared encoding:
```cpp
auto memDescTy = dyn_cast<MemDescType>(operand.getType());
auto shape = memDescTy.getShape();  // or getAllocShape().take_back(rank)
auto encoding = memDescTy.getEncoding();
auto ll = toLinearLayout(shape, encoding);
// Then:
auto offsetBases = flattenBases(ll.getBases().lookup(kOffset));
auto blockBases  = flattenBases(ll.getBases().lookup(kBlock));
```

The existing `flattenBases` lambda (lines 1467-1472) works for any `BasesT` — it just flattens rows of a 2D structure into a 1D `SmallVector<int32_t, 16>`.

### Extracting Alignment

Alignment comes from the encoding attr, not the MemDescType itself. All shared encodings implement `SharedEncodingTrait` (`TritonGPUAttrInterfaces.td:27-36`), which provides `getAlignment()` returning `int32_t` (default 16).

For `SharedLinearEncodingAttr` specifically (`TritonGPUAttrDefs.td:407-421`):
```cpp
int32_t getAlignment() const { return static_cast<int32_t>(getLayoutAlignment()); }
```
The ODS parameter is `"unsigned":$layoutAlignment` (line 407).

**Recommended call path:** `cast<SharedEncodingTrait>(encoding).getAlignment()` — works for all shared encodings uniformly.

## Recommended Approach

### Change 1: ODS Relaxation (`TritonGPUOps.td:803`)

**File:** `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`

Change line 803 from:
```tablegen
    Variadic<TT_Tensor>:$inputs,
```
to:
```tablegen
    Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs,
```

No other ODS changes needed. The `functional-type($inputs, $results)` assembly format already handles mixed types. Rebuilds: `ninja triton-opt` to regenerate tablegen output and recompile MLIR dialect.

### Change 2: SpecInput → Variant (`clang_compiler.cc:1419-1435`)

Replace `SpecInput` with:
```cpp
struct TensorSpecInput {
  std::string dtype;
  llvm::SmallVector<int64_t, 4> shape;
  int64_t numWarps;
  llvm::SmallVector<int32_t, 16> regBases;
  llvm::SmallVector<int32_t, 16> laneBases;
  llvm::SmallVector<int32_t, 16> warpBases;
};

struct SharedSpecInput {
  std::string dtype;
  llvm::SmallVector<int64_t, 4> shape;
  std::string memorySpace;  // always "shared"
  llvm::SmallVector<int32_t, 16> offsetBases;
  llvm::SmallVector<int32_t, 16> blockBases;
  int32_t alignment;
};

// ExternCallSpec::inputs becomes:
llvm::SmallVector<std::variant<TensorSpecInput, SharedSpecInput>, 4> inputs;
```

No new includes needed — `<variant>` is already in `clang_compiler.h`.

### Change 3: extractExternCallSpecs Branching (`clang_compiler.cc:1442-1500`)

Add dim-name constants for shared layouts (alongside existing `kRegister`/`kLane`/`kWarp` at lines 1442-1447):
```cpp
StringAttr kOffset = StringAttr::get(module.getContext(), "offset");
StringAttr kBlock  = StringAttr::get(module.getContext(), "block");
```

The per-operand loop (line 1455) becomes:
```cpp
for (auto operand : op.getInputs()) {
  auto type = operand.getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    // --- existing distributed path (lines 1457-1499) ---
    auto shape = tensorTy.getShape();
    auto encoding = tensorTy.getEncoding();
    auto ll = toLinearLayout(shape, encoding);
    TensorSpecInput input;
    input.shape.assign(shape.begin(), shape.end());
    input.numWarps = ll.getInDimSize(kWarp);
    input.regBases  = flattenBases(ll.getBases().lookup(kRegister));
    input.laneBases = flattenBases(ll.getBases().lookup(kLane));
    input.warpBases = flattenBases(ll.getBases().lookup(kWarp));
    input.dtype = mapDtype(tensorTy.getElementType());
    spec.inputs.push_back(std::move(input));
  } else if (auto memDescTy = dyn_cast<MemDescType>(type)) {
    // --- new shared-memory path ---
    auto shape = memDescTy.getShape();
    auto encoding = memDescTy.getEncoding();
    auto ll = toLinearLayout(memDescTy);  // uses toLinearLayout(MemDescType)
    SharedSpecInput input;
    input.shape.assign(shape.begin(), shape.end());
    input.memorySpace = "shared";
    input.offsetBases = flattenBases(ll.getBases().lookup(kOffset));
    input.blockBases  = flattenBases(ll.getBases().lookup(kBlock));
    auto sharedEnc = cast<SharedEncodingTrait>(encoding);
    input.alignment = sharedEnc.getAlignment();
    input.dtype = mapDtype(memDescTy.getElementType());
    spec.inputs.push_back(std::move(input));
  }
}
```

**Note:** `toLinearLayout(MemDescType)` (overload at `LinearLayoutConversions.cpp:1376`) uses `getAllocShape().take_back(getRank())` rather than `getShape()` — this handles subview/slice offsets correctly. Use that overload, not `toLinearLayout(getShape(), getEncoding())`.

### Change 4: JSON Serialization (`clang_compiler.cc:1529-1560`)

The per-input JSON loop becomes a `std::visit`:
```cpp
for (auto &inputV : spec.inputs) {
  std::visit([&](auto &input) {
    os << "{";
    os << "\"dtype\": \"" << input.dtype << "\", ";
    os << "\"shape\": [...]";
    // Emit fields specific to the variant type:
    if constexpr (/* TensorSpecInput */) {
      os << "\"num_warps\": " << input.numWarps << ", ";
      os << "\"reg_bases\": ..., \"lane_bases\": ..., \"warp_bases\": ...";
    } else {
      os << "\"memory_space\": \"" << input.memorySpace << "\", ";
      os << "\"offset_bases\": ..., \"block_bases\": ..., \"alignment\": ...";
    }
    os << "}";
  }, inputV);
}
```

**Discriminator strategy:** The shared variant includes `"memory_space": "shared"`, absent from the tensor variant. Phase 6's consumer checks for `"memory_space"` to decide whether to build a `llvm.TensorParameter` or `llvm.SharedTensorParameter`.

## Test Strategy

### Lit Test Location and Convention

**Directory:** `test/TritonGPU/` — standard location for TritonGPU dialect lit tests.
**RUN line pattern** (from `test/TritonGPU/invalid-attributes.mlir:1`):
```
// RUN: triton-opt %s -split-input-file -verify-diagnostics
```

**Build dependency:** Lit tests require `triton-opt` rebuild after ODS changes: `cd $BUILD_DIR && ninja triton-opt`

### Test 1: Mixed Tensor+MemDesc Parse (SHMLIR-01)

**File:** `test/TritonGPU/extern-call-mixed-inputs.mlir`

```mlir
// RUN: triton-opt %s -split-input-file -verify-diagnostics

// Test: ttg.extern_call with mixed TT_Tensor + TTG_MemDescType inputs
// passes verification (no type constraint error).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared  = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0], [2, 0]]}, alignment = 16>

tt.func @mixed_extern_call(%tensor: tensor<32x64xf32, #blocked>,
                           %memdesc: !ttg.memdesc<32x64xf32, #shared, #ttg.shared_memory>) {
  // CHECK: ttg.extern_call
  %result = ttg.extern_call(%tensor, %memdesc)
    { symbol = "test_fn", libpath = "test.cu" }
    : (tensor<32x64xf32, #blocked>, !ttg.memdesc<32x64xf32, #shared, #ttg.shared_memory>) -> tensor<32x64xf32, #blocked>
  tt.return
}
```

### Test 2: Tensor-Only Regression (SHMLIR-01)

**File:** `test/TritonGPU/extern-call-tensor-only.mlir`

```mlir
// RUN: triton-opt %s -split-input-file -verify-diagnostics

// Test: tensor-only extern_call still works after ODS relaxation.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>

tt.func @tensor_only_extern_call(%lhs: tensor<32x64xf32, #blocked>,
                                 %rhs: tensor<32x64xf32, #blocked>) {
  // CHECK: ttg.extern_call
  %result = ttg.extern_call(%lhs, %rhs)
    { symbol = "elementwise_add", libpath = "tt_plugin.cu" }
    : (tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #blocked>
  tt.return
}
```

**Note:** These tests use `-verify-diagnostics` to check parse succeeds (no expected-error markers). Alternatively, use `triton-opt %s | FileCheck %s` to verify the round-trip parse→print.

## Pitfalls & Constraints

### Pitfall 1: Tablegen Rebuild After ODS Change
**What goes wrong:** Modifying `.td` files invalidates generated C++ but cmake may not detect it. Running `ninja` without a clean tablegen rebuild can produce stale or incorrect generated headers.
**How to avoid:** After changing `TritonGPUOps.td`, run: `cd $BUILD_DIR && ninja triton-opt` — this triggers the tablegen re-run via cmake dependencies.
**Warning signs:** Odd compilation errors about undefined types or missing `Variadic` specializations.

### Pitfall 2: `toLinearLayout(MemDescType)` vs `toLinearLayout(shape, encoding)`
**What goes wrong:** Calling `toLinearLayout(memDescTy.getShape(), encoding)` directly for a memdesc with subviews (from `memdesc_index`/`memdesc_subslice`) produces the wrong layout because the shape has been reduced but the allocation shape is larger.
**How to avoid:** Always use the `toLinearLayout(MemDescType)` overload (`LinearLayoutConversions.cpp:1376-1383`) which correctly calls `getAllocShape().take_back(getRank())`.
**Warning signs:** Shared layout extraction produces dimension mismatches.

### Pitfall 3: Shared Layouts Have No `kWarp` Dim
**What goes wrong:** Attempting `ll.getInDimSize(kWarp)` or `ll.getBases().lookup(kWarp)` on a shared LinearLayout will assert/return empty because shared layouts only have `"offset"` and `"block"` input dims.
**How to avoid:** The two code paths (tensor vs memdesc) are separate — never use `kWarp`/`kRegister`/`kLane` in the shared branch.
**Warning signs:** Assertion failure in `LinearLayout::getInDimSize()`.

### Pitfall 4: Variadic Operand Assumption in Downstream Passes
**What goes wrong:** Per STATE.md: "any downstream pass that assumes tensor-only inputs must be verified." A pass doing `for (auto opEl : op->getOperands()) { auto t = cast<RankedTensorType>(opEl.getType()); }` will crash on mixed inputs.
**Mitigation:** Phase 5 scope is limited to ODS + extractExternCallSpecs. The two downstream passes that interact with `extern_call` operands are:
- `_pre_compile_extern_calls` (Phase 6) — will be updated with shared-input handling
- `ExternCallOpToLLVM.cpp` (Phase 6) — will lower memdesc inputs to `ptr addrspace(3)`
Neither is touched in Phase 5. The ODS relaxation ensures parse-time type safety; any crash post-parse requires a `dyn_cast` in the downstream pass (which Phase 6 will handle).

### Pitfall 5: Missing `std::variant` in C++17 mode
**What goes wrong:** If the build is configured with C++14, `std::variant` won't compile.
**Verified:** `clang_compiler.h:36` already `#include <variant>` and uses `std::variant` extensively (lines 164, 188, 319, 402, 406) — the project builds with C++17.
**No action needed.**

### Pitfall 6: `SharedEncodingTrait` is an Interface, Not a Base Class
**What goes wrong:** `cast<SharedEncodingTrait>(encoding)` works (it's an MLIR interface), but `SharedEncodingTrait` is abstract. The actual getter is on each concrete encoding attr (e.g., `SharedLinearEncodingAttr::getAlignment()`).
**Verified:** `SharedEncodingTrait` defines `getAlignment()` with a default implementation returning 16 (`TritonGPUAttrInterfaces.td:34-35`). `cast<SharedEncodingTrait>(encoding).getAlignment()` is correct — it dispatches via the interface.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mixed-type op operands | Custom type constraint trait | `AnyTypeOf<[TT_Tensor, TTG_MemDescType]>` from OpBase.td | MLIR-idiomatic, standard trait, verifier auto-generated |
| Shared layout linearization | Custom layout-to-basis conversion | `toLinearLayout(MemDescType)` from LinearLayoutConversions.cpp | Already handles all 4 shared encoding variants (Swizzled, SharedLinear, NVMMA, AMDRotating) |
| Alignment extraction | Hardcoded or per-encoding switch | `SharedEncodingTrait::getAlignment()` | Interface method — works uniformly across all shared encodings |
| JSON serialization | Hand-written string utility | `llvm::raw_string_ostream` (already used) | Existing pattern in `tritonExtractExternCallSpecs()` — no new library needed |
| Variant dispatch | `if/else` with `std::holds_alternative` | `std::visit` | Type-safe, exhaustive — compiler warns on missing cases |

**Key insight:** The shared-layout extraction path is a near mirror of the distributed-layout path. The only differences are (a) `dyn_cast<MemDescType>` vs `dyn_cast<RankedTensorType>`, (b) dim names `"offset"`/`"block"` vs `"register"`/`"lane"`/`"warp"`, (c) `getAlignment()` instead of `getInDimSize(kWarp)`. Do not build a new abstraction layer — reuse the existing `flattenBases` lambda and `toLinearLayout` infrastructure.

## Package Legitimacy Audit

> This phase modifies the in-tree build system and ODS/C++ source files only. No external packages are installed. No audit needed.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| cmake + ninja | ODS tablegen rebuild + C++ compilation | ✓ | cmake 3.x, ninja | — |
| clang++ (host) | Compiling clang_compiler.cc | ✓ | system clang | — |
| triton-opt | Lit tests (parse verification) | Builds from source | 3.0.0 | — |
| FileCheck | Lit tests | ✓ | via lit | — |

**Missing dependencies with no fallback:** None. All dependencies are in-project or system-provided.

## Validation Architecture

> `workflow.nyquist_validation` not present in `.planning/config.json` (absent = enabled).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | lit (LLVM Integrated Tester) |
| Config file | `test/lit.cfg.py`, `test/lit.site.cfg.py.in` |
| Quick run command | `cd $BUILD_DIR && ninja triton-opt && lit -v test/TritonGPU/extern-call-mixed-inputs.mlir` |
| Full suite command | `cd $BUILD_DIR && ninja triton-opt && lit -v test/TritonGPU/` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SHMLIR-01 | Mixed tensor+memdesc extern_call passes verification | lit (parse) | `lit -v test/TritonGPU/extern-call-mixed-inputs.mlir` | ❌ Wave 0 |
| SHMLIR-01 | Tensor-only extern_call still passes verification | lit (parse) | `lit -v test/TritonGPU/extern-call-tensor-only.mlir` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** Build (`ninja triton-opt`) + relevant lit tests
- **Per wave merge:** Full TritonGPU lit suite
- **Phase gate:** Both new lit tests green

### Wave 0 Gaps
- [ ] `test/TritonGPU/extern-call-mixed-inputs.mlir` — covers SHMLIR-01 mixed parse
- [ ] `test/TritonGPU/extern-call-tensor-only.mlir` — covers SHMLIR-01 tensor regression

## Security Domain

> No `security_enforcement` key in `.planning/config.json`. Treat as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | — |
| V3 Session Management | No | — |
| V4 Access Control | No | — |
| V5 Input Validation | Yes | ODS type constraints at parse time prevent invalid-type operands; `dyn_cast` with defaults prevents unexpected-type crashes |
| V6 Cryptography | No | — |

### Known Threat Patterns for MLIR/C++

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Blind `cast<>` (crash on unexpected type) | Denial of Service | `dyn_cast` with fallback (per D-10) |
| Missing operands → out-of-bounds access | Tampering | `SameVariadicOperandSize` trait verifies operand count at parse time |
| Unvalidated JSON output → consumer parsing errors | Tampering | Discriminator key (`memory_space`) clearly separates variants |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `AnyTypeOf` from `mlir/IR/OpBase.td` supports `Variadic<AnyTypeOf<[...]>>` — works for Variadic operands, not just single operands [ASSUMED] | ODS Relaxation | Low — `mlir/IR/OpBase.td` is a standard MLIR dependency; `AnyTypeOf` is a `TypeConstraint` used by both `Variadic` and non-Variadic operands. Pattern precedent in `TritonOps.td:877`. |
| A2 | The `toLinearLayout(MemDescType)` overload at `LinearLayoutConversions.cpp:1376` correctly handles subview-adjusted shapes [ASSUMED] | Shared Layout Extraction | Medium — if the overload produces incorrect offsets for subviewed memdescs, Phase 7 E2E tests will catch it. Phase 5 lit tests don't exercise subviews. |
| A3 | `SharedEncodingTrait::getAlignment()` default of 16 is correct for all shared encodings in this phase [ASSUMED] | Shared Layout Extraction | Low — the interface default is 16, individual encodings override. `SharedLinearEncodingAttr` provides its own `getAlignment()` returning `getLayoutAlignment()`. |

## Open Questions

1. **Should the lit tests use `-verify-diagnostics` or `FileCheck`?**
   - What we know: `-verify-diagnostics` confirms parse succeeds (no errors). `FileCheck` can verify round-trip output but is overkill for parse-only validation per D-11.
   - What's unclear: Whether the test should verify the printed MLIR round-trips correctly (e.g., that `AnyTypeOf` operands don't get dropped/reordered during print).
   - Recommendation: Use `-verify-diagnostics` for parse-only (per D-11). Add a `FileCheck` round-trip test if there's concern about the assembly format with mixed types.

2. **What MLIR syntax exactly for the memdesc operand?**
   - What we know: MemDescType syntax is `!ttg.memdesc<dimsxT, #encoding, #memory_space, [mutable]>` per `TritonGPUTypes.td:82` (`hasCustomAssemblyFormat = 1`). The memory space for shared is `#ttg.shared_memory`.
   - What's unclear: Whether `ttg.extern_call`'s `functional-type($inputs, $results)` printer correctly handles the memdesc type syntax (custom format).
   - Recommendation: Test with `triton-opt` on the proposed lit test MLIR to confirm parsing round-trips before locking the test syntax.

3. **JSON discriminator: `"memory_space"` key vs explicit `"kind"` field?**
   - What we know: D-10 specifies `SharedSpecInput::memorySpace = "shared"`. The tensor variant has no memory_space field.
   - What's unclear: Whether Phase 6 consumer should check for the presence of `"memory_space"` key (implicit discriminator) or we should add an explicit `"kind": "tensor"`/`"kind": "shared"` field for clarity.
   - Recommendation: Use presence of `"memory_space"` as the implicit discriminator (simpler JSON). Document in the JSON output contract. If ambiguity arises in Phase 6, add an explicit `"kind"` field retroactively.

## Sources

### Primary (HIGH confidence — verified via codebase grep/read)
- `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:786-814` — ODS op definition [VERIFIED: codebase]
- `include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:23-84` — MemDescType ODS [VERIFIED: codebase]
- `python/src/clang_compiler.cc:1417-1567` — SpecInput, extractExternCallSpecs, JSON serialization [VERIFIED: codebase]
- `python/src/clang_compiler.cc:1-47` — includes (variant, LLVM, MLIR Dialect, LinearLayout) [VERIFIED: codebase]
- `python/src/clang_compiler.h:36,164,188` — variant in header [VERIFIED: codebase]
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:1323-1401` — toLinearLayout overloads [VERIFIED: codebase]
- `python/src/gluon_ir.cc:238-242` — shared dim names (offset, block) [VERIFIED: codebase]
- `lib/Dialect/TritonGPU/IR/Dialect.cpp:3940-3964` — getSharedLayoutStr confirms offset/block dims [VERIFIED: codebase]
- `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:395-426` — SharedLinearEncodingAttr [VERIFIED: codebase]
- `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td:27-36` — SharedEncodingTrait with getAlignment() [VERIFIED: codebase]
- `third_party/nvidia/backend/compiler.py:709-867` — _pre_compile_extern_calls consumer [VERIFIED: codebase]
- `include/triton/Dialect/Triton/IR/TritonOps.td:877` — AnyTypeOf precedent [VERIFIED: codebase]
- `test/TritonGPU/invalid-attributes.mlir:1` — lit test RUN line convention [VERIFIED: codebase]
- `test/TritonGPU/ops.mlir:89,142-143` — shared_linear MLIR syntax examples [VERIFIED: codebase]
- `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:1-30` — includes, TTG_Op base class [VERIFIED: codebase]
- `.planning/phases/04-c-templates-clang-ast-foundation/04-CONTEXT.md:1-50` — Phase 4 decisions [VERIFIED: codebase]
- `.planning/config.json` — workflow config (nyquist enabled) [VERIFIED: codebase]
- `lib/Dialect/TritonGPU/IR/Dialect.cpp:2061-2070` — SharedLinearEncodingAttr::toLinearLayout implementation [VERIFIED: codebase]

### Secondary (MEDIUM confidence)
- `.planning/phases/05-mlir-op-relaxation-spec-extraction/05-CONTEXT.md` — User decisions (D-09 through D-11) [CITED]
- `.planning/REQUIREMENTS.md` — SHMLIR-01, SHMLIR-02 [CITED]
- `.planning/ROADMAP.md` — Phase 5 scope, Phase 4 dependency [CITED]
- `.planning/STATE.md` — Blocker: Variadic blast radius downstream pass concern [CITED]

### Tertiary (LOW confidence — assumptions)
- A1: AnyTypeOf support for Variadic
- A2: toLinearLayout(MemDescType) correctness for subviewed shapes
- A3: Default alignment via SharedEncodingTrait

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no external libraries; all changes are in-project
- Architecture: HIGH — existing infrastructure (toLinearLayout, std::variant, flattenBases) fully supports the needed extensions
- Pitfalls: HIGH — identified from code inspection of all relevant codepaths
- Shared layout extraction: HIGH — verified dim names, API, and alignment extraction against 3 source files

**Research date:** 2026-07-15
**Valid until:** 2026-08-15 (stable MLIR infrastructure, changes are 1-2 source files)
