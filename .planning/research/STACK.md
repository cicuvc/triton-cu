# Technology Stack: v1.1 Shared Memory Interop for `gl.call()`

**Project:** triton-cu — in-process CUDA C++ interop
**Milestone:** v1.1 — Shared-memory argument passing via `SharedTensor<dtype, shape, layout>&`
**Researched:** 2026-07-12
**Overall confidence:** HIGH (every claim grounded in `file:line` in this repo)

## Executive Summary

The v1.1 milestone adds shared-memory (`addrspace 3`) argument passing to `gl.call()`. The implementation must extend the existing v1.0 `TensorParameter`/`TypeBuilder`/`TypeInspector`/`FunctionResolver` pipeline to handle a new C++ `SharedTensor` device-side type backed by a `SharedLinearLayout` representation. On the MLIR side, shared memory is already modeled as a `ttg::MemDescType` with `SharedMemorySpaceAttr` and lowered to LLVM `ptr addrspace(3)` via `SharedMemoryObject` — the infrastructure exists and needs to be wired into the extern-call lowering path.

**Critical correction to AGENTS.md:** The claim that "shared memory buffers are represented as memref" is wrong. Shared memory uses Triton's custom `ttg::MemDescType` (TritonGPUTypes.td:23), NOT MLIR's `memref`. It lowers to `LLVM::LLVMPointerType` with address space 3 (`getSharedAddressSpace()`), not to any `memref` LLVM type. The `SharedMemoryObject` utility class wraps addrspace-3 base pointers + offsets.

---

## 1. Shared-Memory MLIR Types and Encodings

### 1.1 The Memory Descriptor Type

The concrete MLIR type for `shared_memory_descriptor` is:

```
!ttg.memdesc<shape, elementType, encoding, #ttg.shared_memory, mutableMemory, allocShape>
```

| Component | Value for shared memory | Defined at |
|-----------|------------------------|------------|
| Type class | `ttg::MemDescType` | `include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:23-84` |
| `memorySpace` parameter | `ttg::SharedMemorySpaceAttr` | `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1496-1501` |
| `mutableMemory` | `true` (for writable shared memory) | set by `get_shared_mem_desc_ty` at `gluon_ir.cc:315` |
| `allocShape` | user-specified allocation shape | `_layouts.py` `SharedLinearLayout.shape` is the logical shape; alloc_shape may be larger for padding |

**Build path (Python → C++ → MLIR):**
1. `shared_memory_descriptor_type.to_ir(builder)` → `gluon_ir.cc:308-318`:
   ```cpp
   // gluon_ir.cc:308-318
   .def("get_shared_mem_desc_ty",
        [](GluonOpBuilder &self, Type &elementType,
           std::vector<int64_t> &shape, Attribute layout,
           std::vector<int64_t> &allocShape) -> Type {
          auto ctx = self.getContext();
          return self.getChecked<ttg::MemDescType>(
              shape, elementType, layout,
              ttg::SharedMemorySpaceAttr::get(ctx),
              /*mutableMemory=*/true,
              /*allocShape=*/allocShape);
        })
   ```

### 1.2 Shared Memory Encodings (Layout Attributes)

The encoding attribute on a `MemDescType` determines the address mapping from logical tensor indices to shared memory byte offsets. The relevant encodings:

| Encoding | MLIR Attribute | Purpose | Defined at |
|----------|---------------|---------|------------|
| **SharedLinearEncoding** | `#ttg.shared_linear<{linearLayout, alignment}>` | Explicit linear bases (offset + block) with alignment — the target encoding for v1.1 | `TritonGPUAttrDefs.td:395-426` |
| SwizzledSharedEncoding | `#ttg.swizzled_shared<{vec, perPhase, maxPhase, order, CGALayout}>` | Traditional bank-conflict-avoiding swizzle; used for most shared memory | `TritonGPUAttrDefs.td:10-206` |
| NVMMASharedEncoding | `#ttg.nvmma_shared<{swizzlingByteWidth, transposed, elementBitWidth, fp4Padded, CGALayout}>` | MMAv3/MMAv5 (Hopper/Blackwell) shared memory tile layout | `TritonGPUAttrDefs.td:428-496` |
| PaddedSharedEncoding | `#ttg.padded_shared<{intervals, paddings, linearComponent}>` | Wraps another shared encoding with padding for bank conflict avoidance | `TritonGPUAttrDefs.td:208-326` |
| PartitionedSharedEncoding | `#ttg.partitioned_shared` | Multi-partition layout (dynamic warp group indexing) | `TritonGPUAttrDefs.td:330-381` |

**v1.1 target:** `SharedLinearEncodingAttr`. It is the most general shared memory encoding — `offset_bases` + `block_bases` + `alignment` fully describe the address mapping for arbitrary swizzle patterns:

```cpp
// TritonGPUAttrDefs.td:395-426
def SharedLinearEncodingAttr
    : TritonGPU_Attr<"SharedLinearEncoding", "shared_linear_encoding",
                     [SharedEncodingTrait, LayoutEncodingTrait,
                      DeclareLayoutEncodingMethods]> {
  let parameters = (ins LinearLayoutParam:$linearLayout, "unsigned":$layoutAlignment);
  // getAlignment() returns static_cast<int32_t>(getLayoutAlignment()) — triton line 421
}
```

### 1.3 SharedEncodingTrait and LayoutEncodingTrait Interfaces

All shared encodings implement `SharedEncodingTrait` (TritonGPUAttrInterfaces.td:27). The key method is:

```cpp
int32_t getAlignment() const;  // via DeclareSharedEncodingMethods, TritonGPUAttrInterfaces.td:38-39
```

`LayoutEncodingTrait` / `DeclareLayoutEncodingMethods` provides `toLinearLayout(shape) → LinearLayout` which returns the linear address mapping for the encoding (Dialect.cpp:2061-2070 for SharedLinearEncoding).

### 1.4 Python ↔ C++ ↔ MLIR Mapping

| Python class | MLIR encoding attribute | Builder method |
|-------------|------------------------|----------------|
| `SharedLinearLayout` (`_layouts.py:631`) | `SharedLinearEncodingAttr` | `builder.get_shared_linear_layout()` at `gluon_ir.cc:454-466` |
| `PaddedSharedLayout` (`_layouts.py:556`) | `PaddedSharedEncodingAttr` | `builder.get_padded_shared_layout()` at `gluon_ir.cc:440-453` |
| (new) `SharedLinearLayout` C++ template | n/a (clang AST) | Will mirror `TypeBuilder::BuildLayout` but with offset/block bases |

**`get_shared_linear_layout` implementation** (`gluon_ir.cc:454-466`):
```cpp
.def("get_shared_linear_layout",
     [](GluonOpBuilder &self, std::vector<std::vector<int>> &offsetBases,
        std::vector<std::vector<int>> &blockBases,
        unsigned alignment) -> Attribute {
       auto ctx = self.getContext();
       auto kOffset = mlir::StringAttr::get(ctx, "offset");
       auto kBlock = mlir::StringAttr::get(ctx, "block");
       auto outDims = tt::standardOutDimNames(ctx, offsetBases[0].size());
       auto ll = tt::LinearLayout(
           {{kOffset, offsetBases}, {kBlock, blockBases}}, outDims);
       return self.getChecked<ttg::SharedLinearEncodingAttr>(
           ctx, std::move(ll), alignment);
     })
```

---

## 2. Shared Memory Lowering to LLVM (Address Space 3)

### 2.1 Correction: "memref" vs "MemDesc"

**AGENTS.md line 15 is wrong.** Shared memory is NOT represented as MLIR's `memref` type. The accurate chain is:

```
MLIR: ttg::MemDescType (TritonGPUTypes.td:23)
  → Lowering: SharedMemoryObject (Utility.h:374)
    → LLVM: ptr addrspace(3) (via getSharedMemoryBase, Utility.cpp:1569-1589)
```

`SharedMemoryObject` is NOT an MLIR `memref` — it is a plain C++ utility struct wrapping:

```cpp
// Utility.h:374-448
class SharedMemoryObject {
  SmallVector<Value> bases;    // LLVM ptr addrspace(3) values
  Type baseElemType;
  SmallVector<Value> offsets;  // i32 values, zero at allocation
};
```

### 2.2 Shared Memory Allocation and Address-Space Assignment

The pipeline for shared memory address-space assignment:

1. **AllocateSharedMemory pass** (`AllocateSharedMemory.cpp:17-26`):
   - Analyzes all `ttg::LocalAllocOp` (and other memdesc-producing ops)
   - Runs `ModuleAllocation` analysis → assigns `allocation.offset` IntegerAttr per op
   - For partitioned tensors: assigns `allocation.offset` ArrayAttr (one per partition)

2. **`getSharedMemoryBase()`** (`Utility.cpp:1569-1589`):
   ```cpp
   Value getSharedMemoryBase(Location loc, RewriterBase &rewriter,
                             const TargetInfoBase &target, Operation *op) {
     auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                             target.getSharedAddressSpace());  // = 3
     // ... reads allocation.offset attr, GEP from stack pointer
     Value base = b.gep(ptrTy, i8_ty, LLVM::getStackPointer(rewriter, func), offVal);
     return base;
   }
   ```

3. **NVIDIA `getSharedAddressSpace()`** returns **3** always:
   ```cpp
   // third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.cpp:620
   int TargetInfo::getSharedAddressSpace() const { return 3; }
   ```

### 2.3 Load/Store from Shared Memory

`materializeLocalAddrs()` at `Utility.cpp:616-658` demonstrates the pattern:

```cpp
SmallVector<LocalSharedMemoryAddress>
materializeLocalAddrs(Location loc, triton::gpu::MemDescType memDescTy,
                      SharedMemoryObject smemObj, Type llvmElemTy, ...) {
  // ...
  Value affineOffset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
  // ...
  ptr = b.gep(smemObj.getBase().getType(), llvmElemTy, smemObj.getBase(), offset);
  addrs.push_back({ptr, blockId});
  // Then load/store use this ptr addrspace(3) directly
}
```

**Key insight for v1.1 lowering:** When passing a shared memory arg to an extern device function, we need to extract `smemObj.getBase()` (an `ptr addrspace(3)`) and pass it directly to the callee. No alloca/store needed — the shared memory pointer already lives in addrspace 3.

### 2.4 MemDesc Subviews

Operations like `ttg.memdesc_subslice`, `ttg.memdesc_trans`, `ttg.memdesc_reshape`, `ttg.memdesc_reinterpret` create view-type memdesc ops without copying data. The `SharedMemoryObject` tracks accumulated offsets. For extern-call shared memory passing, we should pass the base pointer of the resolved `SharedMemoryObject`, not the original allocation's base.

---

## 3. Clang/LLVM APIs for Shared-Memory Pointers in Device Functions

### 3.1 CUDA C++ Side: New Templates

The existing `tt_plugin.cu` defines:
- `Shape<DIMS...>` (line 8)
- `TensorLayout<TShape, N_WARPS>` with `BasisGroup`, `Layout` (lines 31-79)
- `Tensor<T, TShape, TLayout>` (lines 85-99)
- `PlaceholderLayout` (lines 81-83)

**v1.1 requires new C++ templates:**

```cpp
// New in tt_plugin.cu: shared-memory layout representation
template<typename TShape, uint32_t N_WARPS>
struct SharedLinearLayout {
    static constexpr uint32_t RANK = TShape::RANK;
    // offset_bases: bases mapping from shared memory offset bits to logical indices
    // block_bases: bases mapping from block-ID bits to logical indices
    // alignment: shared memory alignment constraint
};
```

Should mirror the existing `Layout` template structure but with offset/block bases instead of reg/lane/warp bases.

### 3.2 Clang AST Construction (`TypeBuilder`)

`TypeBuilder` (`clang_compiler.h:224-257`) currently builds distributed tensor types:
- `BuildLayoutFactory(shape, N_WARPS)` → `BuildBasisGroup()` × 3 (reg/lane/warp) → `BuildLayout()` → `BuildTensor()`

**v1.1 needs a parallel path** from a new `SharedTensorParameter` struct:
- **New struct:** `SharedTensorParameter { ScalarType Type; vector<uint32_t> Shape; SharedLayoutInfo Layout; }`
- **New `SharedLayoutInfo` struct:** `{ vector<uint32_t> OffsetBases; vector<uint32_t> BlockBases; uint32_t Alignment; }`
- **LValue reference:** `BuildTensor()` returns `clang::QualType`. To get `SharedTensor<T,Shape,SL>&`, prepend `clang::LValueReferenceType`:
  ```cpp
  auto tensorTy = helper.Builder.BuildTensor(elementTy, shapeTy, layoutTy);
  return Ctx.getLValueReferenceType(tensorTy);
  ```

### 3.3 Clang AST Parsing (`TypeInspector`)

`TypeInspector` (`clang_compiler.h:263-280`) reverses the AST → `TensorParameter`. Currently `ParseTensorType()` extracts scalar type, shape, and `ParseLayoutType()` extracts reg/lane/warp bases.

**v1.1 needs:**
- `TypeInspector::ParseSharedLayoutType()` — parse `SharedLinearLayout<offset_bases, block_bases, alignment>` into `SharedLayoutInfo`
- `TypeInspector::ParseSharedTensorType()` — handle `SharedTensor<T, Shape, SharedLinearLayout>`
- `DispatchTypeParsing()` must recognize the `SharedTensor` template and dispatch accordingly
- `FunctionResolver::LookupFunction()` must handle `SharedTensor` reference args — the existing Sema template deduction should work natively for the new templates

### 3.4 LLVM IR for `__device__` Functions with Shared Memory Args

When clang compiles a `__device__` function taking a reference to shared memory:
```cpp
__device__ void foo(SharedTensor<float, Shape<32>, MyLayout>& smem) { ... }
```

Clang's CodeGen naturally produces:
```
define dso_local void @_Z3foo...(
    ptr addrspace(3) %smem  // <-- reference = ptr in addrspace 3
) {
```

**No special clang API or attribute is needed** — the `__device__` function compiled with `--cuda-device-only` automatically puts shared-memory-pointed-to parameters in addrspace 3. The parameter type in the LLVM IR will be `ptr addrspace(3)`, matching what `getSharedMemoryBase()` produces.

### 3.5 Integration with ExternCallOpToLLVM Lowering

The existing lowering (`ExternCallOpToLLVM.cpp:111-293`) handles tensor args by `alloca + store + ptr` (lines 143-152). For shared memory args, the approach is different:

1. **Identify shared-memory operands:** The `ExternCallOp` needs to distinguish tensor vs. memdesc inputs. Currently the op signature is `Variadic<TT_Tensor>:$inputs` (TritonGPUOps.td:803) — this must be extended.
2. **Extract `SharedMemoryObject`:** Call `getSharedMemoryObjectFromStruct()` (Utility.h:454-457) to unpack the LLVM struct
3. **Pass base pointer directly:** `smemObj.getBase()` is already `ptr addrspace(3)` — pass it as a callee arg directly (no alloca/store)
4. **Match callee signature:** The callee's LLVM function should have `ptr addrspace(3)` for the shared-memory parameter position

---

## 4. LLVM Version and Toolchain Compatibility

| Component | Version/Detail | Source |
|-----------|---------------|--------|
| LLVM | 23.0.0git (pre-23 development) | `cmake/llvm-info.json:2` (hash `62b7cf9623fc310525f39ed69aaecc318a909731`); version confirmed at `/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/llvm-data/install/lib/cmake/llvm/LLVMConfigVersion.cmake` — `set(PACKAGE_VERSION "23.0.0git")` |
| Clang resources | `lib/clang/23` | `compiler.py:428` (`_resource_dir`), `compiler.py:736` |
| CUDA Toolkit | 13.1 | `compiler.py:432` (`/usr/local/cuda-13.1/targets/x86_64-linux/include`) |
| Shared address space | `3` (hardcoded) | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.cpp:620` |
| System compiler | `clang` (not gcc) | AGENTS.md:9 (`CC=clang CXX=clang++`) |
| `-fno-rtti` | Required for `clang_compiler.cc` | AGENTS.md:12, `CMakeLists.txt` |

**Compatibility notes:**
- Address space 3 for shared memory is part of the NVPTX backend and has been stable since LLVM 15+; no breaking changes expected.
- The `LLVM::LLVMPointerType::get(ctx, addressSpace)` API is available in LLVM 23.0.0git.
- clang CodeGen for `__device__` functions correctly assigns addrspace 3 to shared-memory-pointed-to parameters — this is standard NVPTX target behavior, not version-specific.
- CUDA 13.1 headers may change the internal structure of `__shared__` variable emission, but since we're _passing_ shared memory pointers (not allocating them inside the device function), this is not a concern.

---

## 5. Integration Points with Existing v1.0 Machinery

### 5.1 Must Extend (Breaking Changes to Existing Code)

| Integration point | File:Line | Current behavior | Required change for v1.1 |
|-------------------|-----------|-----------------|--------------------------|
| `extern_call` op inputs | `TritonGPUOps.td:803` | `Variadic<TT_Tensor>:$inputs` | Must also accept `TTG_MemDescType` — either add mixed-input support or a separate operands field |
| `call_extern()` arg validation | `_semantic.py:254` | `isinstance(a, ttgl.tensor)` — rejects `shared_memory_descriptor` | Allow `shared_memory_descriptor` args; pass them through to `create_extern_call` |
| `extractExternCallSpecs()` | `clang_compiler.cc:1169-1170` | `cast<RankedTensorType>(operand.getType())` — crashes on `MemDescType` | Handle `MemDescType` operands: extract `toLinearLayout()`, emit different JSON spec with `is_shared: true` |
| `_pre_compile_extern_calls()` | `compiler.py:786-795` | Builds `TensorParameter` from `SpecInput` JSON | Build `SharedTensorParameter` for shared-mem specs; pass to clang |

### 5.2 Must Extend (Additive Changes — New Code Paths)

| Integration point | File:Line | Required addition |
|-------------------|-----------|-------------------|
| `CudaFuncRequest` | `clang_compiler.h:165-168` | Add `SharedTensorParameter` to `ParamTypes` variant or a parallel `SharedParamTypes` field |
| `TensorParameter` / `LayoutInfo` | `clang_compiler.h:129-141` | Add `SharedTensorParameter` / `SharedLayoutInfo` structs alongside existing ones |
| `TypeBuilder` | `clang_compiler.h:224-257` | Add `BuildSharedLinearLayout()` + `BuildSharedTensor()` methods; `BuildTensor()` already handles the tensor wrapper, may be reusable for `SharedTensor` |
| `TypeInspector` | `clang_compiler.h:263-280` | Add `ParseSharedLayoutType()`, `ParseSharedTensorType()`; extend `DispatchTypeParsing()` |
| `FunctionResolver` | `clang_compiler.h:286-296` | `LookupFunction()` already takes `ArrayRef<QualType>` — works with `SharedTensor` types if clang sees the template. May need to handle `LValueReferenceType` unwrapping in arg type construction. |
| `CUDACompiler::BuildTensor()` | `clang_compiler.cc:740-767` | Add `BUildSharedTensor()` that constructs `SharedTensor<T,Shape,SharedLinearLayout>` via clang AST |
| `ExternCallOpToLLVM` lowering | `ExternCallOpToLLVM.cpp:111-293` | Detect memdesc operands, extract `SharedMemoryObject::getBase()` (ptr addrspace 3), pass as direct callee arg (no alloca/store) |
| C++ device templates | `tt_plugin.cu` | Add `SharedLinearLayout` template struct + `SharedTensor<T,Shape,SharedLinearLayout>` template struct, modeled on existing `Layout`/`Tensor` |

### 5.3 Must NOT Change (Stable v1.0 Seams)

| Stable API | Reason |
|------------|--------|
| `linkBitcodeToModule()` / `link_cuda_bitcode` | Bitcode linking is type-agnostic; shared-memory callee functions link the same way |
| `tritonCompileCuda()` / `compile_cuda_to_module` | High-level API doesn't change — only the `CudaFuncRequest` data within it |
| `InferExternCallResult` / `infer_extern_call_result` hook | Inference returns `[(scalar_name, result_shape)]` — shared-memory args don't affect return types |
| `tritonPatchExternCallResultTypes()` | Return type patching remains result-only; shared memory is argument-only in v1.1 |
| `layoutToGluon()` / `to_linear_layout` | These handle the distributed-Layout-to-MLIR round-trip; shared memory uses `SharedLinearLayout` → `SharedLinearEncodingAttr` which already exists |
| `codegen_fns` hook pattern | The `infer_extern_call_result` hook registration mechanism is unchanged |
| LLVMContext sharing | `_shared_ctx` reuse between semantic and llir stages stays the same (`compiler.py:620-621`) |

---

## 6. What NOT to Add (Over-Engineering to Avoid)

| Anti-pattern | Why avoid | What to do instead |
|-------------|-----------|-------------------|
| MLIR's `memref` type | Triton uses its own `MemDescType`; mixing the two would require converting between fundamentally different type systems | Use `ttg::MemDescType` consistently; the `SharedMemoryObject` utility already handles the lowering |
| A new MLIR dialect op for extern-call-with-shared-memory | Fragments the `gl.call()` code path; creates wasted duplication | Extend `ttg.extern_call` to accept mixed tensor/memdesc operands |
| Wrapper functions to convert shared-memory pointers | Adds indirection and potential performance cost | Pass `SharedMemoryObject::getBase()` (ptr addrspace 3) directly as callee arg |
| Replacing `SharedMemoryObject` with a new abstraction | `SharedMemoryObject` is the established utility across all Triton shared-memory lowering | Use `getSharedMemoryObjectFromStruct()` / `getBase()` — they already work correctly |
| Building a new AST type system parallel to `TypeBuilder`/`TypeInspector` | Creates a maintenance burden; the existing architecture handles template type construction generically | Extend `TypeBuilder`/`TypeInspector` with new methods for `SharedTensor`/`SharedLinearLayout`; share the Shape/Layout factory pattern |
| `result_layout` derivation from shared-memory layout | Shared-memory args don't produce return types; this is a return-type concern (deferred) | Return-type inference stays unchanged for v1.1 |

---

## 7. LLVM NVPTX Address Space Conventions

| Address Space | NVPTX Name | Used for |
|---------------|------------|----------|
| 0 | `generic` (or default) | Generic pointers, stack locals |
| 1 | `global` | Global memory (device DRAM) |
| 3 | `shared` | Shared memory (on-chip SRAM) |
| 4 | `constant` | Constant memory |
| 5 | `local` | Local (per-thread) memory |

All shared memory pointers in Triton's generated LLVM IR use address space 3. This is enforced by `getSharedMemoryBase()` at `Utility.cpp:1571-1572` which queries `target.getSharedAddressSpace()` and uses it to construct the pointer type. The NVIDIA target returns 3 at `TargetInfo.cpp:620`.

**Implication for extern calls:** When clang compiles a `__device__` function taking `SharedTensor<T,Shape,SL>&`, the parameter becomes `ptr addrspace(3)` in LLVM IR. The caller (Triton lowering) passes `smemObj.getBase()` which is also `ptr addrspace(3)`. The types match natively — no address-space casting is needed.

---

## Sources (confidence-tagged)

| Source | Location | What it provides | Confidence |
|--------|----------|-----------------|------------|
| ODS type definition | `TritonGPUTypes.td:23-84` | `MemDescType` with Shape, elementType, encoding, memorySpace, mutableMemory, allocShape | HIGH (ODS is canonical) |
| ODS shared memory space | `TritonGPUAttrDefs.td:1496-1501` | `SharedMemorySpaceAttr` → `#ttg.shared_memory` | HIGH (ODS) |
| ODS shared linear encoding | `TritonGPUAttrDefs.td:395-426` | `SharedLinearEncodingAttr` with LinearLayout + alignment | HIGH (ODS) |
| Python shared linear layout | `_layouts.py:631-673` | `SharedLinearLayout` dataclass mapping to `SharedLinearEncodingAttr` | HIGH (existing, tested) |
| Shared descriptor type build | `gluon_ir.cc:308-318` | `get_shared_mem_desc_ty` → `MemDescType` + `SharedMemorySpaceAttr` | HIGH (existing, tested) |
| Shared linear layout build | `gluon_ir.cc:454-466` | `get_shared_linear_layout` → `SharedLinearEncodingAttr` | HIGH (existing, tested) |
| Shared memory base utility | `Utility.cpp:1569-1589` | `getSharedMemoryBase()` → `ptr addrspace(3)` via GEP | HIGH (core infrastructure) |
| SharedMemoryObject class | `Utility.h:374-448` | Wraps addrspace-3 bases + offsets | HIGH (core infrastructure) |
| NVIDIA target addrspace | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.cpp:620` | `getSharedAddressSpace() returns 3` | HIGH (canonical) |
| AllocateSharedMemory pass | `AllocateSharedMemory.cpp:17-26` | Assigns `allocation.offset` attributes | HIGH (required pipeline pass) |
| ExternCallOp definition | `TritonGPUOps.td:786-814` | `Variadic<TT_Tensor>:$inputs` (must be extended) | HIGH (ODS) |
| Extern call lowering | `ExternCallOpToLLVM.cpp:111-293` | By-pointer arg convention, alloca+store pattern | HIGH (existing, tested) |
| Spec extraction | `clang_compiler.cc:1152-1219` | `extractExternCallSpecs` — `cast<RankedTensorType>` (must be extended) | HIGH (existing, tested) |
| TypeBuilder/TensorParameter | `clang_compiler.h:129-141,224-257` | AST construction from TensorParameter | HIGH (v1.0-validated) |
| TypeInspector | `clang_compiler.h:263-280` | Reverse AST → TensorParameter | HIGH (v1.0-validated) |
| LLVM version | `/media/.../LLVMConfigVersion.cmake` | `PACKAGE_VERSION "23.0.0git"` | HIGH (build artifact) |
| CUDA C++ templates | `tt_plugin.cu:1-99` | Existing Shape, TensorLayout, Tensor, PlaceholderLayout | HIGH (existing, tested) |
| Semantic arg validation | `_semantic.py:254` | `isinstance(a, ttgl.tensor)` — rejects memdesc | HIGH (existing code) |
| Pre-compile extern calls | `compiler.py:709-807` | Builds TensorParameter from JSON specs | HIGH (existing, tested) |
| LocalAllocOp definition | `TritonGPUOps.td:152-192` | Produces `MemDescType` result, checks `isSharedMemoryAlloc()` | HIGH (ODS) |
| toLinearLayout for MemDesc | `LinearLayoutConversions.cpp:1376-1392` | LinearLayout from MemDescType (uses allocShape) | HIGH (core conversion) |
