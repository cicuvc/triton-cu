# Phase 5: MLIR Op Relaxation + Spec Extraction - Pattern Map

**Mapped:** 2026-07-15
**Files analyzed:** 5 (2 modified, 2 created, 1 read-only reference)
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:803` | model (ODS tablegen) | request-response (parse-time type enforcement) | `include/triton/Dialect/Triton/IR/TritonOps.td:877` | exact |
| `python/src/clang_compiler.cc:1419-1567` | service (spec extraction + serialization) | transform (MLIR → JSON) | `python/src/clang_compiler.cc:1438-1505` (existing distributed path — self-analog) | exact |
| `python/src/clang_compiler.h` | model (header types) | shared data model | `python/src/clang_compiler.h:36` (existing `std::variant` includes) | exact |
| `test/TritonGPU/extern-call-mixed-inputs.mlir` | test (MLIR parse lit) | request-response (parse verify) | `test/TritonGPU/invalid-attributes.mlir` + `test/TritonGPU/ops.mlir:89` | exact |
| `test/TritonGPU/extern-call-tensor-only.mlir` | test (MLIR parse lit) | request-response (parse verify) | `test/TritonGPU/invalid-attributes.mlir` (RUN line pattern) | exact |
| `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` | N/A (read-only reference) | reference | N/A | N/A |

## Pattern Assignments

---

### 1. `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:803` (model, request-response)

**Analog:** `include/triton/Dialect/Triton/IR/TritonOps.td:877` — the only existing `Variadic<AnyTypeOf<[...]>>` pattern in the project's TableGen files.

**Existing code to modify** (`TritonGPUOps.td:786-814`):
```tablegen
def TTG_ExternCallOp : TTG_Op<"extern_call", [
    SameVariadicOperandSize
  ]> {
  let arguments = (ins
    Variadic<TT_Tensor>:$inputs,         // ← LINE 803: replace this line
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

**Analog pattern** (`TritonOps.td:877`):
```tablegen
  let arguments = (ins StrAttr:$asm_string, StrAttr:$constraints, BoolAttr:$pure, I32Attr:$packed_element, Variadic<AnyTypeOf<[TT_Type]>>:$args);
```

**Change — line 803 becomes:**
```tablegen
    Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs,
```

**Includes already present** (`TritonGPUOps.td:1-14`):
```tablegen
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"     // line 6 — provides TTG_MemDescType
include "triton/Dialect/Triton/IR/TritonTypes.td"            // line 11 — provides TT_Tensor
include "mlir/IR/OpBase.td"                                   // line 14 — provides AnyTypeOf
```

**Verification impact:** The `TTG_Op` base class (`TritonGPUOps.td:27-29`) auto-appends `VerifyTensorLayoutsTrait` + `VerifyMemDescLayoutsTrait`. `SameVariadicOperandSize` validates operand count. The `AnyTypeOf` constraint enforces parse-time types. No verifier changes needed. Assembly format `functional-type($inputs, $results)` already handles mixed types.

**Notes:** Same pattern also at `TritonOps.td:954` (second usage in `TT_DebugBarrierOp`). The `TT_Type` is the Triton dialect's AnyTypeOf constraint; `TT_Tensor` and `TTG_MemDescType` are the specific types for this op.

---

### 2. `python/src/clang_compiler.cc:1419-1435` — SpecInput → Variant Replacement (SHMLIR-02, D-10)

**Analog:** The existing `SpecInput` struct in the same file. This is a self-analog — the struct is being replaced with a variant.

**Existing structs** (`clang_compiler.cc:1419-1435`):
```cpp
namespace {

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

} // namespace
```

**Replace with:**
```cpp
namespace {

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

struct ExternCallSpec {
  std::string symbol;
  std::string libpath;
  bool useFastMath = false;
  llvm::SmallVector<std::variant<TensorSpecInput, SharedSpecInput>, 4> inputs;
};

} // namespace
```

**`std::variant` already included** (`clang_compiler.h:36`):
```cpp
#include <variant>
```

---

### 3. `python/src/clang_compiler.cc:1438-1505` — extractExternCallSpecs Branching

**Analog:** The existing distributed-layout extraction path (`clang_compiler.cc:1442-1500`) — self-analog. The shared-memory branch mirrors this pattern with different dim names and alignment extraction.

**Existing per-operand loop** (`clang_compiler.cc:1442-1500`):
```cpp
  StringAttr kRegister =
      StringAttr::get(module.getContext(), "register");
  StringAttr kLane =
      StringAttr::get(module.getContext(), "lane");
  StringAttr kWarp =
      StringAttr::get(module.getContext(), "warp");

  module.walk([&](mlir::triton::gpu::ExternCallOp op) {
    ExternCallSpec spec;
    spec.symbol = op.getSymbol().str();
    spec.libpath = op.getLibpath().str();
    spec.useFastMath = op.getUseFastMath();

    for (auto operand : op.getInputs()) {
      auto tensorTy = cast<RankedTensorType>(operand.getType());  // ← LINE 1456: changes to dyn_cast
      auto shape = tensorTy.getShape();
      auto encoding = tensorTy.getEncoding();

      auto ll =
          mlir::triton::gpu::toLinearLayout(shape, encoding);

      SpecInput input;
      input.shape.assign(shape.begin(), shape.end());
      input.numWarps = ll.getInDimSize(kWarp);

      auto flattenBases = [](auto bases) {          // line 1467 — reused as-is
        llvm::SmallVector<int32_t, 16> flat;
        for (auto &row : bases)
          flat.append(row.begin(), row.end());
        return flat;
      };

      input.regBases =
          flattenBases(ll.getBases().lookup(kRegister));
      input.laneBases =
          flattenBases(ll.getBases().lookup(kLane));
      input.warpBases =
          flattenBases(ll.getBases().lookup(kWarp));

      auto elemTy = tensorTy.getElementType();
      if (isa<Float32Type>(elemTy))             // line 1482 — shared with shared path
        input.dtype = "f32";
      else if (isa<Float64Type>(elemTy))
        input.dtype = "f64";
      else if (isa<Float16Type>(elemTy))
        input.dtype = "f16";
      else if (isa<BFloat16Type>(elemTy))
        input.dtype = "bf16";
      else if (elemTy.isInteger(32))
        input.dtype = "i32";
      else if (elemTy.isInteger(64))
        input.dtype = "i64";
      else if (elemTy.isInteger(8))
        input.dtype = "i8";
      else
        input.dtype = "f32";

      spec.inputs.push_back(std::move(input));
    }
    results.push_back(std::move(spec));
  });
```

**New dim-name constants** (alongside existing `kRegister`/`kLane`/`kWarp` at lines 1442-1447):
```cpp
  StringAttr kRegister =
      StringAttr::get(module.getContext(), "register");
  StringAttr kLane =
      StringAttr::get(module.getContext(), "lane");
  StringAttr kWarp =
      StringAttr::get(module.getContext(), "warp");
  StringAttr kOffset =                                    // ← NEW
      StringAttr::get(module.getContext(), "offset");
  StringAttr kBlock =                                     // ← NEW
      StringAttr::get(module.getContext(), "block");
```

**Dim-name source:** Confirmed by `gluon_ir.cc:238-242`:
```cpp
    auto kOffset = mlir::StringAttr::get(ctx, "offset");
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    return layouts.SharedLinearLayout(
        toStdVector(ll.getBases().lookup(kOffset)),
        toStdVector(ll.getBases().lookup(kBlock)), sharedLl.getAlignment());
```

**New per-operand branching** (replaces lines 1455-1500):
```cpp
    for (auto operand : op.getInputs()) {
      auto type = operand.getType();
      if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
        // --- existing distributed path (unchanged) ---
        auto shape = tensorTy.getShape();
        auto encoding = tensorTy.getEncoding();
        auto ll = mlir::triton::gpu::toLinearLayout(shape, encoding);

        TensorSpecInput input;
        input.shape.assign(shape.begin(), shape.end());
        input.numWarps = ll.getInDimSize(kWarp);

        auto flattenBases = [](auto bases) {
          llvm::SmallVector<int32_t, 16> flat;
          for (auto &row : bases)
            flat.append(row.begin(), row.end());
          return flat;
        };

        input.regBases  = flattenBases(ll.getBases().lookup(kRegister));
        input.laneBases = flattenBases(ll.getBases().lookup(kLane));
        input.warpBases = flattenBases(ll.getBases().lookup(kWarp));

        auto elemTy = tensorTy.getElementType();
        if (isa<Float32Type>(elemTy)) input.dtype = "f32";
        else if (isa<Float64Type>(elemTy)) input.dtype = "f64";
        else if (isa<Float16Type>(elemTy)) input.dtype = "f16";
        else if (isa<BFloat16Type>(elemTy)) input.dtype = "bf16";
        else if (elemTy.isInteger(32)) input.dtype = "i32";
        else if (elemTy.isInteger(64)) input.dtype = "i64";
        else if (elemTy.isInteger(8)) input.dtype = "i8";
        else input.dtype = "f32";

        spec.inputs.push_back(std::move(input));
      } else if (auto memDescTy = dyn_cast<MemDescType>(type)) {
        // --- new shared-memory path ---
        auto shape = memDescTy.getShape();
        auto encoding = memDescTy.getEncoding();
        // Use MemDescType overload to correctly handle subview-adjusted alloc shapes
        auto ll = mlir::triton::gpu::toLinearLayout(memDescTy);

        SharedSpecInput input;
        input.shape.assign(shape.begin(), shape.end());
        input.memorySpace = "shared";

        auto flattenBases = [](auto bases) {
          llvm::SmallVector<int32_t, 16> flat;
          for (auto &row : bases)
            flat.append(row.begin(), row.end());
          return flat;
        };

        input.offsetBases = flattenBases(ll.getBases().lookup(kOffset));
        input.blockBases  = flattenBases(ll.getBases().lookup(kBlock));

        auto sharedEnc = cast<SharedEncodingTrait>(encoding);
        input.alignment = sharedEnc.getAlignment();

        auto elemTy = memDescTy.getElementType();
        if (isa<Float32Type>(elemTy)) input.dtype = "f32";
        else if (isa<Float64Type>(elemTy)) input.dtype = "f64";
        else if (isa<Float16Type>(elemTy)) input.dtype = "f16";
        else if (isa<BFloat16Type>(elemTy)) input.dtype = "bf16";
        else if (elemTy.isInteger(32)) input.dtype = "i32";
        else if (elemTy.isInteger(64)) input.dtype = "i64";
        else if (elemTy.isInteger(8)) input.dtype = "i8";
        else input.dtype = "f32";

        spec.inputs.push_back(std::move(input));
      }
    }
```

**Key API references for the shared path:**

`toLinearLayout(MemDescType)` overload (`LinearLayoutConversions.cpp:1376-1383`):
```cpp
LinearLayout toLinearLayout(MemDescType type) {
  auto shape = type.getAllocShape().take_back(type.getRank());
  return toLinearLayout(shape, type.getEncoding());
}
```

`SharedEncodingTrait::getAlignment()` (`TritonGPUAttrInterfaces.td:27-36`):
```tablegen
def SharedEncodingTrait : AttrInterface<"SharedEncodingTrait"> {
  let methods = [
    InterfaceMethod<"Return the default alignment for the layout.",
                    "int32_t", "getAlignment", (ins), [{}], [{ return 16; }]>,
  ];
}
```

**`MemDescType` API** (`TritonGPUTypes.td:33-40`):
```
getShape() → ArrayRef<int64_t>     // line 34
getElementType() → Type             // line 35
getEncoding() → Attribute           // line 36
getMemorySpace() → Attribute        // line 37
getMutableMemory() → bool           // line 38
getAllocShape() → ArrayRef<int64_t> // line 39
```

---

### 4. `python/src/clang_compiler.cc:1514-1567` — JSON Serialization (std::visit)

**Analog:** The existing `tritonExtractExternCallSpecs()` function (`clang_compiler.cc:1514-1567`) — self-analog. The JSON loop is replaced with a `std::visit` over the variant.

**Existing JSON serialization** (`clang_compiler.cc:1514-1567`):
```cpp
std::string
tritonExtractExternCallSpecs(mlir::ModuleOp module) {
  auto specs = extractExternCallSpecs(module);

  std::string jsonStr;
  llvm::raw_string_ostream os(jsonStr);
  os << "[";
  bool firstSpec = true;
  for (auto &spec : specs) {
    if (!firstSpec) os << ", ";
    firstSpec = false;
    os << "{";
    os << "\"symbol\": \"" << spec.symbol << "\", ";
    os << "\"libpath\": \"" << spec.libpath << "\", ";
    os << "\"use_fast_math\": " << (spec.useFastMath ? "true" : "false") << ", ";
    os << "\"inputs\": [";
    bool firstInput = true;
    for (auto &input : spec.inputs) {         // ← LINE 1531: becomes std::visit
      if (!firstInput) os << ", ";
      firstInput = false;
      os << "{";
      os << "\"dtype\": \"" << input.dtype << "\", ";
      os << "\"shape\": [";
      for (size_t i = 0; i < input.shape.size(); ++i) {
        if (i > 0) os << ", ";
        os << input.shape[i];
      }
      os << "], ";
      os << "\"num_warps\": " << input.numWarps << ", ";
      auto flatten = [&](auto &bases) {
        os << "[";
        for (size_t i = 0; i < bases.size(); ++i) {
          if (i > 0) os << ", ";
          os << bases[i];
        }
        os << "]";
      };
      os << "\"reg_bases\": ";
      flatten(input.regBases);
      os << ", \"lane_bases\": ";
      flatten(input.laneBases);
      os << ", \"warp_bases\": ";
      flatten(input.warpBases);
      os << "}";
    }
    os << "]}";
  }
  os << "]";
  os.flush();
  return jsonStr;
}
```

**New std::visit per-input loop** (replaces lines 1531-1561):
```cpp
    for (auto &inputV : spec.inputs) {
      if (!firstInput) os << ", ";
      firstInput = false;
      std::visit([&](auto &input) {
        os << "{";
        os << "\"dtype\": \"" << input.dtype << "\", ";
        os << "\"shape\": [";
        for (size_t i = 0; i < input.shape.size(); ++i) {
          if (i > 0) os << ", ";
          os << input.shape[i];
        }
        os << "]";

        if constexpr (std::is_same_v<std::decay_t<decltype(input)>, TensorSpecInput>) {
          // Tensor variant: emit distributed layout fields
          os << ", \"num_warps\": " << input.numWarps;
          auto flatten = [&](auto &bases) {
            os << "[";
            for (size_t i = 0; i < bases.size(); ++i) {
              if (i > 0) os << ", ";
              os << bases[i];
            }
            os << "]";
          };
          os << ", \"reg_bases\": ";
          flatten(input.regBases);
          os << ", \"lane_bases\": ";
          flatten(input.laneBases);
          os << ", \"warp_bases\": ";
          flatten(input.warpBases);
        } else {
          // Shared variant: emit shared memory layout fields
          os << ", \"memory_space\": \"" << input.memorySpace << "\"";
          auto flatten = [&](auto &bases) {
            os << "[";
            for (size_t i = 0; i < bases.size(); ++i) {
              if (i > 0) os << ", ";
              os << bases[i];
            }
            os << "]";
          };
          os << ", \"offset_bases\": ";
          flatten(input.offsetBases);
          os << ", \"block_bases\": ";
          flatten(input.blockBases);
          os << ", \"alignment\": " << input.alignment;
        }
        os << "}";
      }, inputV);
    }
```

**JSON discriminator strategy:** The shared variant includes `"memory_space": "shared"` (absent from tensor variant). Phase 6 consumer (`compiler.py:786-795`) uses `inp.get("memory_space")` to decide whether to build `TensorParameter` or `SharedTensorParameter`.

**Compatibility:** Phase 5 does NOT modify `compiler.py:786-795`. The existing consumer uses `.get()` with defaults:
```python
for inp in spec_entry["inputs"]:
    tp = llvm.TensorParameter()
    tp.type = _scalar_type_for(inp["dtype"])
    tp.shape = inp["shape"]
    tp.reg_basis = inp.get("reg_bases", [])   # ← default [] — shared inputs won't crash here
    tp.lane_basis = inp.get("lane_bases", [])
    tp.warp_basis = inp.get("warp_bases", [])
    tp.n_warps = inp.get("num_warps", 1)
```
New shared-input JSON keys (`memory_space`, `offset_bases`, `block_bases`, `alignment`) are ignored by the Phase 5 consumer since it only accesses `dtype`/`shape`/`reg_bases`/`lane_bases`/`warp_bases`/`num_warps` with defaults. No crash.

---

### 5. `test/TritonGPU/extern-call-mixed-inputs.mlir` — New Lit Test (SHMLIR-01)

**Analog 1 (RUN line pattern):** `test/TritonGPU/invalid-attributes.mlir:1`
```mlir
// RUN: triton-opt %s -split-input-file -verify-diagnostics
```

**Analog 2 (memdesc syntax):** `test/TritonGPU/ops.mlir:89,142-143`
```mlir
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0], [4, 0]], block = [[2, 0]]}, alignment = 16>
#smem = #ttg.shared_memory
// Usage: !ttg.memdesc<8x16xf32, #shared, #smem>

#shared_linear_16 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 4], [4, 8], [8, 0]]}, alignment = 512>
```

**Analog 3 (blocked encoding for tensor):** `test/TritonGPU/invalid-attributes.mlir:4`
```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
```

**Test file (`test/TritonGPU/extern-call-mixed-inputs.mlir`):**
```mlir
// RUN: triton-opt %s -split-input-file -verify-diagnostics

// Test: ttg.extern_call with mixed TT_Tensor + TTG_MemDescType inputs
// passes verification (no type constraint error).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared  = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0], [2, 0]], block = [[2, 0]]}, alignment = 16>
#smem    = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_extern_call(%tensor: tensor<32x64xf32, #blocked>,
                             %memdesc: !ttg.memdesc<32x64xf32, #shared, #smem>) {
    // CHECK: ttg.extern_call
    %result = ttg.extern_call(%tensor, %memdesc)
      { symbol = "test_fn", libpath = "test.cu" }
      : (tensor<32x64xf32, #blocked>, !ttg.memdesc<32x64xf32, #shared, #smem>) -> tensor<32x64xf32, #blocked>
    tt.return
  }
}
```

**Notes:** Uses `-verify-diagnostics` — no `expected-error` markers means parse must succeed. Module wrapping required (`tt.func` inside `module` with target attributes) — matches TritonGPU lit test convention (cf. `ops.mlir:91`).

---

### 6. `test/TritonGPU/extern-call-tensor-only.mlir` — New Lit Test (SHMLIR-01 regression)

**Analog:** Same RUN line pattern as above. Tensor-only syntax unchanged from the existing extern_call ODS description.

**Test file (`test/TritonGPU/extern-call-tensor-only.mlir`):**
```mlir
// RUN: triton-opt %s -split-input-file -verify-diagnostics

// Test: tensor-only extern_call still works after ODS relaxation.
// Validates the AnyTypeOf constraint does not reject pure tensor operands.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tensor_only_extern_call(%lhs: tensor<32x64xf32, #blocked>,
                                   %rhs: tensor<32x64xf32, #blocked>) {
    // CHECK: ttg.extern_call
    %result = ttg.extern_call(%lhs, %rhs)
      { symbol = "elementwise_add", libpath = "tt_plugin.cu" }
      : (tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #blocked>
    tt.return
  }
}
```

---

### 7. `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` — Read-Only Reference (No Modifications)

**Purpose:** Reference only — `extractExternCallSpecs()` calls `toLinearLayout(memDescTy)` which dispatches to the overload at line 1376.

**Key overload** (`LinearLayoutConversions.cpp:1376-1383`):
```cpp
LinearLayout toLinearLayout(MemDescType type) {
  // Pass in the allocation shape. Then when using invertAndCompose it will
  // trim the allocationShape to the shape if they are different.
  auto shape = type.getAllocShape().take_back(type.getRank());
  return toLinearLayout(shape, type.getEncoding());
}
```

**Other overloads for reference** (`LinearLayoutConversions.cpp:1372-1392`):
```cpp
LinearLayout toLinearLayout(RankedTensorType type) {
  return toLinearLayout(type.getShape(), type.getEncoding());
}

LinearLayout toLinearLayout(TensorOrMemDesc type) {
  if (auto ranked = dyn_cast<RankedTensorType>(type)) {
    return toLinearLayout(ranked);
  } else {
    auto memDesc = cast<MemDescType>(type);
    return toLinearLayout(memDesc);
  }
}
```

---

## Shared Patterns

### Pattern 1: `AnyTypeOf` ODS Type Constraint
**Source:** `include/triton/Dialect/Triton/IR/TritonOps.td:877`
**Apply to:** `TritonGPUOps.td:803`
```tablegen
Variadic<AnyTypeOf<[TT_Type]>>:$args
```
For extern_call, use specific types: `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs`

### Pattern 2: `flattenBases` Lambda
**Source:** `clang_compiler.cc:1467-1472`
**Apply to:** Both distributed and shared branches in `extractExternCallSpecs()`
```cpp
auto flattenBases = [](auto bases) {
  llvm::SmallVector<int32_t, 16> flat;
  for (auto &row : bases)
    flat.append(row.begin(), row.end());
  return flat;
};
```

### Pattern 3: Dtype String Mapping
**Source:** `clang_compiler.cc:1481-1497`
**Apply to:** Both distributed and shared branches — identical logic, can be extracted to a lambda:
```cpp
auto mapDtype = [](mlir::Type elemTy) -> std::string {
  if (isa<Float32Type>(elemTy))  return "f32";
  if (isa<Float64Type>(elemTy))  return "f64";
  if (isa<Float16Type>(elemTy))  return "f16";
  if (isa<BFloat16Type>(elemTy)) return "bf16";
  if (elemTy.isInteger(32))      return "i32";
  if (elemTy.isInteger(64))      return "i64";
  if (elemTy.isInteger(8))       return "i8";
  return "f32";
};
```

### Pattern 4: Lit Test RUN Line
**Source:** `test/TritonGPU/invalid-attributes.mlir:1`
**Apply to:** All new lit tests
```mlir
// RUN: triton-opt %s -split-input-file -verify-diagnostics
```

### Pattern 5: `llvm::raw_string_ostream` JSON
**Source:** `clang_compiler.cc:1517-1565`
**Apply to:** JSON serialization of the variant — same `llvm::raw_string_ostream` pattern, `std::visit` over the variant to emit type-specific fields.

### Pattern 6: MLIR Module Wrapping for Lit Tests
**Source:** `test/TritonGPU/ops.mlir:91`
**Apply to:** Both new lit tests
```mlir
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @name(...) {
    ...
    tt.return
  }
}
```

### Pattern 7: `SharedEncodingTrait` Interface for Alignment
**Source:** `TritonGPUAttrInterfaces.td:27-36`
**Apply to:** Shared-spec extraction branch
```cpp
auto sharedEnc = cast<SharedEncodingTrait>(encoding);
input.alignment = sharedEnc.getAlignment();
// Default: 16. Individual encodings (SharedLinearEncodingAttr, SwizzledSharedEncodingAttr, etc.) override.
```

### Pattern 8: Shared Layout Dim Names ("offset", "block")
**Source:** `gluon_ir.cc:238-242`, `lib/Dialect/TritonGPU/IR/Dialect.cpp:3947-3948`
**Apply to:** Shared-spec extraction branch
```cpp
StringAttr kOffset = StringAttr::get(module.getContext(), "offset");
StringAttr kBlock  = StringAttr::get(module.getContext(), "block");
// NEVER use kRegister/kLane/kWarp for shared layouts — they don't have those dims
```

---

## No Analog Found

No files with zero matches. All files have exact or self-analog matches.

---

## Metadata

**Analog search scope:** `include/triton/Dialect/*/IR/*.td`, `python/src/clang_compiler.{cc,h}`, `test/TritonGPU/*.mlir`, `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`, `python/src/gluon_ir.cc`, `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td`
**Files scanned:** 12
**Pattern extraction date:** 2026-07-15
