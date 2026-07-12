# Architecture: Shared Memory Interop for `gl.call()` (v1.1)

**Domain:** Compiler internals — Triton MLIR/LLVM/Clang interop extension
**Date:** 2026-07-12
**Builds on:** `.planning/research/STACK.md`, `.planning/codebase/ARCHITECTURE.md`
**Downstream:** Phase ordering in `.planning/research/SUMMARY.md` (must stay consistent)

---

## 1. System Overview: Six-Layer Pipeline with Shared Memory Seams

The existing v1.0 `gl.call()` pipeline (documented in `.planning/codebase/ARCHITECTURE.md`) is a six-layer compilation stack. v1.1 extends each layer at a well-defined seam. The diagram below uses "→ NEW:" annotations for every new code path.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 1: Python Gluon Frontend                                  │
│  _core.py:775  gl.call(src_path, func, *args, result_layout, ...)                       │
│  ┌─ L803:  _semantic.to_tensor(a)  — for ALL args currently                            │
│  │  → NEW: must handle shared_memory_descriptor pass-through (not a tensor)             │
│  ├─ _semantic.py:250  call_extern(src_path, func, args, result_layouts, ...)           │
│  │  → NEW L254: isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor))             │
│  │  → NEW L270-278: arg_params extended with "is_shared": True + layout info            │
│  │  → L312-316: result_ir_types + arg_handles → create_extern_call __unchanged__       │
│  └─ _core.py:291  shared_memory_descriptor(handle, elTy, shape, layout, alloc_shape)   │
│     _.handle is the MLIR Value for the memdesc (already compatible with op building)     │
└───────────────────────┬────────────────────────────────────────────────────────────────┘
                        │ builder.create_extern_call(libpath, symbol, arg_handles, resultTypes)
                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 2: MLIR Op Layer                                          │
│  TritonGPUOps.td:786-814  ttg.extern_call                                                │
│  ┌─ L802-808 CURRENT:  Variadic<TT_Tensor>:$inputs, StrAttr:$symbol, StrAttr:$libpath  │
│  │  → NEW: Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs                    │
│  │         (or a NEW $shared_inputs Variadic<TTG_MemDescType> for cleaner dispatch)      │
│  └─ gluon_ir.cc:615-624  create_extern_call()                                            │
│     → Passes args to ExternCallOp::create() — ODS change propagates automatically       │
│     → No C++ change needed here if ODS handles both types correctly                     │
└───────────────────────┬────────────────────────────────────────────────────────────────┘
                        │ MLIR module walk + operand type dispatch
                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 3: Spec Extraction (MLIR → JSON)                          │
│  clang_compiler.cc:1152-1219  extractExternCallSpecs()                                   │
│  ┌─ L1169-1170:  for (auto operand : op.getInputs())                                    │
│  │               auto tensorTy = cast<RankedTensorType>(operand.getType());              │
│  │  → NEW: if (auto memdescTy = dyn_cast<MemDescType>(operand.getType()))               │
│  │           → extract shared layout info via toLinearLayout(memdescTy)                  │
│  │             ref: LinearLayoutConversions.cpp:1376                                     │
│  │           → SpecInput with is_shared + offset/block bases + alignment                 │
│  │  → FALLBACK: existing RankedTensorType path (L1174-1214) unchanged                   │
│  │                                                                                       │
│  │  SpecInput extension (clang_compiler.cc:1120-1142):                                   │
│  │  → NEW fields: bool isShared, offsetBases, blockBases, alignment                     │
│  │                                                                                       │
│  │  JSON serialization (L1278-1281, tritonExtractExternCallSpecs):                       │
│  │  → NEW: "memory_space": "shared" key for shared inputs                               │
│  │  → NEW: "offset_bases", "block_bases", "alignment" keys                              │
│  │  → Compat: non-shared inputs omit memory_space (backward-compatible JSON)             │
│  └─ Output: JSON array of specs, each spec with mixed tensor/shared inputs              │
└───────────────────────┬────────────────────────────────────────────────────────────────┘
                        │ JSON specs → _pre_compile_extern_calls()
                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 4: CUDA JIT Compilation (Python → Clang AST)              │
│  compiler.py:709-849  _pre_compile_extern_calls()                                        │
│  ┌─ L717:  json_str = llvm.extract_extern_call_specs(mod)                               │
│  │  → NEW: shared-input specs have "memory_space": "shared"                             │
│  │                                                                                       │
│  │  Suspended-compiler path (L771-806):                                                  │
│  │  → NEW L771-773: input loop must branch on "is_shared" / memory_space                │
│  │     → if shared: build llvm.SharedTensorParameter()                                  │
│  │         stp = llvm.SharedTensorParameter()                                            │
│  │         stp.type = _scalar_type_for(inp["dtype"])                                     │
│  │         stp.shape = inp["shape"]                                                      │
│  │         stp.offset_bases = inp["offset_bases"]                                        │
│  │         stp.block_bases = inp["block_bases"]                                          │
│  │         stp.alignment = inp.get("alignment", 16)                                      │
│  │         param_types.append(stp)   # variant now includes SharedTensorParameter       │
│  │     → else: existing TensorParameter path (L786-795) unchanged                       │
│  │                                                                                       │
│  │  Fallback path (L808-832):                                                            │
│  │  → SAME shared-vs-tensor branch at L820-830                                          │
│  └─ Requests dispatched to llvm.compile_cuda_to_module() or hook.compile_bitcode()       │
│                                                                                          │
│  C++ data structures (clang_compiler.h):                                                 │
│  ┌─ NEW SharedLayoutInfo (alongside LayoutInfo at L129-135):                             │
│  │     struct SharedLayoutInfo {                                                         │
│  │       std::vector<uint32_t> OffsetBases;  // flat: [off0_0, off0_1, ..., off1_0, ...]│
│  │       std::vector<uint32_t> BlockBases;   // flat, same layout                        │
│  │       uint32_t Alignment = 16;                                                        │
│  │     };                                                                                │
│  │                                                                                       │
│  ├─ NEW SharedTensorParameter (alongside TensorParameter at L137-141):                   │
│  │     struct SharedTensorParameter {                                                    │
│  │       ScalarType Type;                                                                │
│  │       std::vector<uint32_t> Shape;                                                    │
│  │       SharedLayoutInfo Layout;                                                        │
│  │     };                                                                                │
│  │                                                                                       │
│  └─ NEW CudaFuncRequest extension (L165-169):                                            │
│        std::vector<std::variant<ScalarType, TensorParameter, SharedTensorParameter>>     │
│         ParamTypes;                                                                      │
│                                                                                          │
│  Clang AST construction (TypeBuilder):                                                   │
│  ┌─ NEW BuildSharedLinearLayout(OffsetBases, BlockBases, Alignment) → QualType          │
│  │  Parallel to BuildLayout() (L249-252). Uses an analogous factory pattern:              │
│  │    1. Build the SharedLinearLayout template specialization                            │
│  │    2. offset_bases → BasisGroup<N_OFFSET_AXES>, block_bases → BasisGroup<N_BLOCK_AXES>│
│  │    3. SharedLinearLayout<offsetBG, blockBG, alignment>                                │
│  │                                                                                       │
│  ├─ NEW BuildSharedTensor(elementTy, shapeTy, sharedLayoutTy) → QualType                │
│  │  Parallel to BuildTensor() (L253-256). Uses the existing Tensor template with          │
│  │  SharedLinearLayout as the Layout parameter. If SharedTensor is a distinct template,  │
│  │  uses a new SharedTensorTemplateType member.                                          │
│  │                                                                                       │
│  └─ LValue reference wrapping:                                                           │
│       clang::QualType refTy = Ctx.getLValueReferenceType(sharedTensorTy);                │
│       // Constructed in CUDACompiler::BuildSharedTensor (new method)                      │
│                                                                                          │
│  Clang AST reverse-parsing (TypeInspector):                                              │
│  ┌─ NEW ParseSharedLayoutType(ClassTemplateSpecializationDecl) → SharedLayoutInfo       │
│  │  Parallel to ParseLayoutType() (L273). Extracts:                                      │
│  │    - Template arg 0: OffsetBasisGroup → ParseBasisGroup() → OffsetBases               │
│  │    - Template arg 1: BlockBasisGroup  → ParseBasisGroup() → BlockBases                │
│  │    - Template arg 2: alignment uint32_t                                                │
│  │                                                                                       │
│  └─ NEW DispatchTypeParsing branch:                                                      │
│       if (TD->getName() == "SharedTensor")                                               │
│         return ParseSharedTensorType(TD);                                                 │
│       else if (TD->getName() == "Tensor" || TD->getName() == "tuple")                    │
│         // existing branches (L278-279)                                                  │
│                                                                                          │
│  Function resolution (FunctionResolver):                                                 │
│  ┌─ LookupFunction(Name, ArgumentTypes) (L292-294)                                       │
│  │  → ArgumentTypes already accepts any clang::QualType vector — SharedTensor& works    │
│  │  → LValueReferenceType is transparent to Sema's template deduction                    │
│  │  → clang::Sema::DeduceTemplateArguments handles SharedTensor naturally                 │
│  └─ No FunctionResolver change needed for argument matching                              │
└───────────────────────┬────────────────────────────────────────────────────────────────┘
                        │ bitcode with SharedTensor& callee signature
                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 5: LLVM Lowering (ttg → LLVM dialect)                     │
│  ExternCallOpToLLVM.cpp:111-293  ExternCallOpConversion::matchAndRewrite()               │
│  ┌─ L137-139:  promotedOperands = promoteOperands(loc, op->getOperands(), ...)          │
│  │  → ALREADY works: MLIR operands include both tensor Values and memdesc Values         │
│  │  → promoteOperands() handles both types — no change needed                           │
│  │                                                                                       │
│  ├─ L141-152:  EXISTING: alloca+store+ptr for all numTensorArgs                         │
│  │  → NEW: branch on operand type BEFORE the alloca loop                                │
│  │  → for each operand i in op.getInputs():                                              │
│  │      if (isa<MemDescType>(operand.getType()))                                          │
│  │        // Shared memory arg: extract SharedMemoryObject, pass base directly           │
│  │        auto memdescTy = cast<MemDescType>(operand.getType());                         │
│  │        auto elemTy = typeConverter->convertType(memdescTy.getElementType());           │
│  │        auto smemObj = LLVM::getSharedMemoryObjectFromStruct(                           │
│  │            loc, promotedOperands[i], elemTy, rewriter);                               │
│  │        promotedOperands[i] = smemObj.getBase();  // ptr addrspace(3) ALREADY        │
│  │      else                                                                             │
│  │        // Distributed tensor: keep existing alloca+store path (L143-152)              │
│  │                                                                                       │
│  ├─ L154-156:  promotedTypes ← from promotedOperands                                    │
│  │  → Now contains mixed: ptr addrspace(0) + ptr addrspace(3) — MLIR handles both       │
│  │                                                                                       │
│  ├─ L248-251:  funcType = LLVM::LLVMFunctionType::get(clangReturnType, promotedTypes)   │
│  │  → Callee's LLVM IR has ptr addrspace(3) for shared-mem param (clang-natural)        │
│  │  → Caller's promotedTypes[i] is same addrspace (3) from SharedMemoryObject           │
│  │  →  Types match natively — no address-space cast needed                              │
│  │                                                                                       │
│  └─ L254:  callOp = LLVM::CallOp::create(rewriter, loc, funcOp, promotedOperands)       │
│     → Passes mixed-addrspace args directly                                              │
│                                                                                          │
│  REMINDER — WHY NO ALLOCA FOR SHARED:                                                    │
│  SharedMemoryObject::getBase() returns a pointer ALREADY in addrspace 3 (from            │
│  getSharedMemoryBase() at Utility.cpp:1586-1587, which constructs `ptr addrspace(3)` via │
│  GEP). Wrapping it in alloca would produce addrspace-0 ptr — an LLVM type mismatch with  │
│  the callee's expected ptr addrspace(3).                                                 │
│                                                                                          │
│  Utility.h:374-448  SharedMemoryObject                                                   │
│  ┌─ L397-402:  getBase() → bases[0]  (assert single-base, non-partitioned)              │
│  │  → Field: SmallVector<Value> bases  (addrspace 3 LLVM Values)                         │
│  │  → Field: SmallVector<Value> offsets  (i32, initially zero)                           │
│  └─ Used by: ExternCallOpConversion (new), MemoryOpToLLVM, ViewOpToLLVM, etc.            │
│                                                                                          │
│  Utility.h:454-457  getSharedMemoryObjectFromStruct()                                    │
│  ┌─ L1428-1452:  Reconstructs SharedMemoryObject from LLVM struct                       │
│  │  → Extracts bases (addrspace 3 ptrs) and offsets (i32) from struct fields            │
│  │  → Already handles multi-base (partitioned) tensors                                   │
│  └─ Important: This reads the FULL SharedMemoryObject including subview offsets           │
│     (not just the raw allocation.offset from getSharedMemoryBase())                       │
│                                                                                          │
│  Utility.cpp:1569-1589  getSharedMemoryBase()                                            │
│  ┌─ L1571-1572:  ptrTy with target.getSharedAddressSpace() (= 3 for NVIDIA)              │
│  │  → GEP from stack pointer by allocation.offset                                        │
│  └─ Returns ptr addrspace(3) — the foundation for SharedMemoryObject::getBase()          │
└───────────────────────┬────────────────────────────────────────────────────────────────┘
                        │ LLVM function call with mixed-addrspace args
                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 6: Bitcode Linking (LLVM IR Merge)                         │
│  clang_compiler.cc  linkBitcodeToModule() + compiler.py:make_llir()                      │
│  ┌─ linkBitcodeToModule() → CloneFunctionInto with DifferentModule flag                 │
│  │  → NO CHANGE: bitcode links identically for SharedTensor callee functions             │
│  │  → Callee already in same LLVMContext (parsed from CUDA bitcode)                     │
│  │  → Ret-type fix (alloca+store+load launder) applies to return values, not args       │
│  │  → alwaysinline + O3 fully inlines the callee                                        │
│  └─ compiler.py:387-400  make_llir() → link_cuda_bitcode()                              │
│     → NO CHANGE: the linked bitcode includes SharedTensor& device functions              │
│     → The callee's ptr addrspace(3) param type is preserved through linking             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Shared-Memory Layout Round-Trip

The v1.0 pipeline has a distributed-layout round-trip: `MLIR encoding → TensorParameter → clang AST → TypeInspector → TensorParameter → MLIR encoding`. v1.1 adds an analogous shared-layout round-trip.

### 2.1 Forward Path: MLIR → Clang AST

```
Step 1 — MLIR encoding extraction:
  ttg::MemDescType with SharedLinearEncodingAttr (#ttg.shared_linear)
    → toLinearLayout(memDescType) at LinearLayoutConversions.cpp:1376-1392
    → Returns LinearLayout with "offset" and "block" in-dim names

Step 2 — Flatten to JSON spec:
  extractExternCallSpecs() at clang_compiler.cc:1152
    → offset_bases = flattenBases(ll.getBases().lookup("offset"))
    → block_bases  = flattenBases(ll.getBases().lookup("block"))
    → alignment    = sharedLl.getAlignment()
    → JSON: {"memory_space":"shared","offset_bases":[[...]],"block_bases":[[...]],"alignment":N}

Step 3 — Build SharedTensorParameter in Python:
  _pre_compile_extern_calls() at compiler.py:771
    → stp = llvm.SharedTensorParameter()
    → stp.offset_bases = inp["offset_bases"]   // flat list
    → stp.block_bases  = inp["block_bases"]
    → stp.alignment    = inp.get("alignment", 16)

Step 4 — Construct clang AST types:
  CUDACompiler::BuildSharedTensor(stp)  // NEW method
    → TypeBuilder::BuildSharedLinearLayout(offsetBases, blockBases, alignment)
      → Constructs clang::ClassTemplateSpecializationDecl for SharedLinearLayout<...>
      → Uses BasisGroup template (generic, N_BASES-parameterized) for both offset and block
    → TypeBuilder::BuildSharedTensor(elementTy, shapeTy, sharedLayoutTy)
      → Constructs SharedTensor<T, Shape, SharedLinearLayout>
      → If SharedTensor is NEW template: use SharedTensorTemplateType
      → If Tensor is reused with SharedLinearLayout: use existing TensorTemplateType
    → clang::LValueReferenceType(sharedTensorTy) for SharedTensor<T,Shape,SL>&

Step 5 — Callee function resolution:
  FunctionResolver::LookupFunction(funcName, argTypes)
    → argTypes contains SharedTensor<T,Shape,SL>& alongside Tensor<T,Shape,Layout>&
    → clang Sema DeduceTemplateArguments handles SharedTensor naturally
    → Returns resolved/instantiated FunctionDecl
```

### 2.2 Reverse Path: Clang AST → Python Data (Return-Type Inference Compatibility)

```
Step 1 — Return type inspection:
  CUDACompiler::EvaluateFunctionReturnType(FD)
    → FD->getCallResultType() → QualType (may be SharedTensor for arg types,
      but SharedTensor results are deferred — v1.1 is argument-only)

Step 2 — If SharedTensor recognized in return position (future milestone):
  TypeInspector::DispatchTypeParsing(QualType) at clang_compiler.h:278
    → if TD->getName() == "SharedTensor"
    → ParseSharedTensorType(TD)
      → ParseShapeType() → shape
      → ParseSharedLayoutType() → SharedLayoutInfo { offsetBases, blockBases, alignment }
      → getScalarTypeFromQualType() → ScalarType
      → returns SharedTensorParameter { ScalarType, Shape, SharedLayoutInfo }

For v1.1: TypeInspector only needs to PARSE shared types (for robustness, not yet used
for result inference). The primary use is recognizing shared-type callee parameters so
LookupFunction can match them correctly — handled by Sema naturally.
```

### 2.3 New Cross-Boundary Data Structs (Repeated for Reference)

All in `clang_compiler.h`:

```cpp
// L129-135 — EXISTING
struct LayoutInfo {
  std::vector<uint32_t> LayoutShape;
  std::vector<uint32_t> RegBasis;
  std::vector<uint32_t> LaneBasis;
  std::vector<uint32_t> WarpBasis;
  uint32_t N_WARPS = 0;
};

// NEW — alongside LayoutInfo
struct SharedLayoutInfo {
  std::vector<uint32_t> OffsetBases;  // flat, row-major: [off0[0], off0[1], ..., off1[0], ...]
  std::vector<uint32_t> BlockBases;   // same layout
  uint32_t Alignment = 16;
};

// L137-141 — EXISTING
struct TensorParameter {
  ScalarType Type;
  std::vector<uint32_t> Shape;
  LayoutInfo Layout;
};

// NEW — alongside TensorParameter
struct SharedTensorParameter {
  ScalarType Type;
  std::vector<uint32_t> Shape;
  SharedLayoutInfo Layout;
};

// L165-169 — MODIFIED
struct CudaFuncRequest {
  std::string Symbol;
  std::vector<std::variant<ScalarType, TensorParameter, SharedTensorParameter>> ParamTypes;
  bool UseFastMath = false;
};
```

### 2.4 C++ Templates in `tt_plugin.cu` (New Definitions)

The existing file (`python/test/gluon/tt_plugin.cu`) defines `Shape` (L7-13), `TensorLayout` (L30-79), `PlaceholderLayout` (L81-83), and `Tensor<T, TShape, TLayout>` (L85-99).

New templates needed (modeled on existing patterns):

```cpp
// NEW — SharedLinearLayout: analogous to TensorLayout but with offset/block bases
template<typename TShape, uint32_t N_WARPS>
struct SharedLinearLayout {
    static constexpr uint32_t RANK = TShape::RANK;
    static constexpr uint32_t REG_SIZE = TShape::SIZE / N_WARPS;

    template<uint32_t N_BASES>
    using BasisGroup = typename TensorLayout<TShape, N_WARPS>::template BasisGroup<N_BASES>;

    template<BasisGroup<N_OFFSET_AXES> OFFSETS,
             BasisGroup<N_BLOCK_AXES> BLOCKS,
             uint32_t ALIGN>
    struct Layout {
        static constexpr uint32_t NUM_WARPS = N_WARPS;
        static constexpr uint32_t REG_SIZE = TShape::SIZE / N_WARPS;
        static constexpr uint32_t Alignment = ALIGN;
        static constexpr auto GROUP_OFFSETS = OFFSETS;
        static constexpr auto GROUP_BLOCKS = BLOCKS;
    };
};

// NEW — SharedTensor: analogous to Tensor but for shared memory
//  MAY REUSE the existing Tensor template if SharedLinearLayout is compatible
//  with the Layout template parameter position.
//  If the SharedLinearLayout constraint differs (e.g., no reg/lane/warp evaluate),
//  a separate SharedTensor template is warranted.
template<typename T, typename TShape, typename TSharedLayout>
struct SharedTensor {
    T data[TSharedLayout::REG_SIZE];

    SharedTensor() = default;

    // Read accessor: load element at logical index computed from offset/block bases
    // (future milestones — this device-side indexing needs careful design)
};
```

**Design decision point:** Whether `SharedTensor` reuses the existing `Tensor<T, TShape, SharedLinearLayout>` or is a separate template. Current research recommends a **separate `SharedTensor` template** because:
- `SharedTensor` data is in shared memory (addrspace 3), not registers — the `data[]` array semantics differ
- `SharedLinearLayout` has fundamentally different evaluate() semantics (offset+block bases vs reg+lane+warp bases)
- A separate template allows compile-time dispatch on `__shared__` storage class

---

## 3. Lowering Data-Flow for Mixed Distributed + Shared Args

### 3.1 How `promoteOperands` Works Today

At `ExternCallOpToLLVM.cpp:137-139`:
```cpp
auto promotedOperands = this->getTypeConverter()->promoteOperands(
    loc, op->getOperands(), adaptor.getOperands(), rewriter);
```

This converts ALL operands (both tensor and memdesc) to their LLVM struct representations. For memdesc operands, the LLVM struct is the `SharedMemoryObject` wire format (bases + offsets). This step is type-agnostic and works as-is for mixed operand lists.

### 3.2 Per-Operand Dispatch (NEW Logic)

The current alloca+store loop at L141-152 processes ALL operands uniformly. **The v1.1 change** is to dispatch per operand based on MLIR type:

```
for (unsigned i = 0; i < op.getInputs().size(); ++i) {
  auto operand = op.getInputs()[i];
  if (isa<ttg::MemDescType>(operand.getType())) {
    // ---- SHARED MEMORY ARG ----
    auto memdescTy = cast<ttg::MemDescType>(operand.getType());
    auto elemTy = typeConverter->convertType(memdescTy.getElementType());
    // promotedOperands[i] is the LLVM struct (SharedMemoryObject wire format)
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, promotedOperands[i], elemTy, rewriter);
    promotedOperands[i] = smemObj.getBase();
    // promotedOperands[i] is now ptr addrspace(3) — matches callee signature
  } else {
    // ---- DISTRIBUTED TENSOR ARG (EXISTING PATH) ----
    auto structVal = promotedOperands[i];
    auto structTy = structVal.getType();
    Value one = b.i32_val(1);
    auto *builder = &static_cast<OpBuilder &>(rewriter);
    Value stackPtr = LLVM::AllocaOp::create(
        *builder, loc, ptrTy, structTy, one, 0).getResult();
    b.store(structVal, stackPtr);
    promotedOperands[i] = stackPtr;
    // promotedOperands[i] is ptr addrspace(0) — matches callee signature
  }
}
```

### 3.3 Type Table for Mixed Args

| MLIR operand type | LLVM representation after promote | After this pass | Callee expects | Match? |
|-------------------|----------------------------------|----------------|----------------|--------|
| `tensor<32xf32, #enc>` | `LLVMStruct {[N x f32]}` | `ptr addrspace(0)` (alloca+store) | `ptr` (addrspace 0) | ✓ |
| `ttg.memdesc<32xf32, #ttg.shared_linear<...>, #ttg.shared_memory>` | `LLVMStruct {ptr addr(3), i32, ...}` | `ptr addrspace(3)` (smemObj.getBase()) | `ptr addrspace(3)` | ✓ |

### 3.4 Callee LLVM IR Signature

When clang compiles:
```cpp
__device__ void my_fn(const Tensor<float, Shape<32>, TLayout>& reg_arg,
                       SharedTensor<float, Shape<32>, SLLayout>& smem_arg) {
```

The LLVM IR (via `-emit-llvm`) naturally produces:
```llvm
define dso_local void @_Z5my_fn...(
    ptr %reg_arg,                    ; addrspace 0 by default (generic)
    ptr addrspace(3) %smem_arg       ; addrspace 3 (shared memory)
) {
```

This is clang's standard NVPTX target behavior — shared-memory-referenced parameters get address space 3. **No special annotations or attributes required.**

### 3.5 Tuple Return + Mixed Args

The tuple-return path (L171-233, `extractorNames` branch) already builds `mainArgTypes` from `promotedTypes` (L188-189). With mixed addrspace args, this works identically — the function types naturally reflect the correct address spaces per argument position.

---

## 4. Recommended Build Order (Phase Decomposition)

### Phase 1: C++ Templates + Clang AST Extension (Foundation)
**Rationale:** The C++ `SharedTensor` and `SharedLinearLayout` types are the semantic foundation that every other layer references. The clang AST bridge must be able to construct and parse these types before any MLIR or lowering work can proceed.

**Deliverables:**
- `SharedLinearLayout` template in `tt_plugin.cu` (L7-99 area)
- `SharedTensor<T, Shape, SharedLinearLayout>` template (or extend `Tensor`)
- `SharedLayoutInfo`, `SharedTensorParameter` structs in `clang_compiler.h`
- `CudaFuncRequest::ParamTypes` extended with `SharedTensorParameter` variant
- `TypeBuilder::BuildSharedLinearLayout()` and `BuildSharedTensor()` methods
- `CUDACompiler::BuildSharedTensor()` (new method, parallel to `BuildTensor` at clang_compiler.cc:730-768)
- `TypeInspector::ParseSharedLayoutType()` and `ParseSharedTensorType()` methods
- `DispatchTypeParsing()` branch for `"SharedTensor"` (clang_compiler.h:278)
- Python bindings: `llvm.SharedTensorParameter()` class exposed from `llvm.cc`

**Dependencies:** None (standalone C++ work)

### Phase 2: MLIR extern_call ODS Relaxation + Spec Extraction
**Rationale:** The `extern_call` op must accept memdesc operands before any integration test can pass. The spec extraction must handle mixed types to connect MLIR to CUDA compilation.

**Deliverables:**
- `TritonGPUOps.td:803` — relax `Variadic<TT_Tensor>:$inputs` to accept `MemDescType`
- `clang_compiler.cc:1169-1213` — `extractExternCallSpecs()` branches on `MemDescType`
- `SpecInput` struct extended with `isShared`, `offsetBases`, `blockBases`, `alignment`
- JSON serialization extended with `"memory_space"`, `"offset_bases"`, etc.
- **Lit test:** extern_call with mixed tensor+memdesc inputs (verification passes)

**Dependencies:** Phase 1 (need `SharedLayoutInfo` shape for spec struct)

### Phase 3: CUDA Compilation Wiring (Python ↔ C++)
**Rationale:** Connect the MLIR-side JSON specs to the CUDA compiler. Without this, specs are extracted but never consumed.

**Deliverables:**
- `compiler.py:771-799` — `_pre_compile_extern_calls()` builds `SharedTensorParameter` for shared inputs
- `compiler.py:820-830` — same for fallback path
- `llvm.cc` — `SharedTensorParameter` pybind11 class with `.type`, `.shape`, `.offset_bases`, `.block_bases`, `.alignment`
- `_pre_compile_extern_calls()` dispatches `llvm.SharedTensorParameter` to requests

**Dependencies:** Phase 1, Phase 2

### Phase 4: LLVM Lowering — Shared Memory Arg Pass-Through
**Rationale:** The lowering must pass `ptr addrspace(3)` directly to the callee. Without this, the call site and callee signatures mismatch.

**Deliverables:**
- `ExternCallOpToLLVM.cpp:141-152` — per-operand dispatch (shared vs distributed)
- `getSharedMemoryObjectFromStruct()` invocation for memdesc operands
- `smemObj.getBase()` passed directly as promoted arg (no alloca)
- Tuple-return path tested with mixed args

**Dependencies:** Phase 1, Phase 2 (needs memdesc operands to lower), Phase 3 (needs callee bitcode with correct signature)

### Phase 5: Frontend Integration (User-Facing API)
**Rationale:** The `gl.call()` API must accept `shared_memory_descriptor` arguments. This is lightweight but gates the end-to-end pipeline.

**Deliverables:**
- `_core.py:803` — `_semantic.to_tensor(a)` adjusted for memdesc pass-through
- `_semantic.py:254` — `isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor))` check
- `_semantic.py:270-278` — `arg_params` extended with `"is_shared": True` and shared layout info for inference

**Dependencies:** All prior phases

### Phase 6: E2E Testing + Verification
**Rationale:** Integration tests that exercise the full pipeline with shared-memory read/write through `gl.call()`.

**Deliverables:**
- New test in `python/test/gluon/test_extern_call.py`: shared-memory arg read+write
- Lit test for `ExternCallOpToLLVM` lowering with mixed args
- Regression: all existing extern-call tests pass unchanged

**Dependencies:** All prior phases

### Parallelization Opportunities

```
Phase 1 (C++ templates) ─────┐
                              ├──→ Phase 3 (CUDA wiring) ──→ Phase 4 (lowering) ──→ Phase 5 → Phase 6
Phase 2 (ODS + specs)  ─────┘

Phases 1 and 2 are independent and can proceed in parallel.
Phase 3 requires both 1 and 2.
```

### Highest-Risk Integration Point

**Phase 2: ODS Type Relaxation** (`TritonGPUOps.td:803`). This is the single change with the widest blast radius. Every consumer of `extern_call` ops — the spec extractor, the verifier, the lowering pattern, the IR printer/parser — must handle the relaxed type. An incorrect ODS change can break:
- ALL existing tensor-only `gl.call()` invocations
- MLIR verification (type mismatch on existing ops)
- Lit tests that pattern-match on `extern_call` types

**Mitigation:** Add lit tests for `extern_call` verification with both tensor-only AND mixed-operand IR before any lowering change. Confirm that existing tensor-only ops still verify correctly.

---

## 5. Architectural Anti-Patterns

### Anti-Pattern 1: Pushing Shared Memory Through the alloca+store Path
**Why it's tempting:** The loop at `ExternCallOpToLLVM.cpp:141-152` processes all operands uniformly. Adding a shared-memory branch feels like extra code.
**Why it's wrong:** `alloca` produces `ptr addrspace(0)`. The callee expects `ptr addrspace(3)`. This causes an LLVM verifier error ("call argument type mismatch") or a silent ptxas address-space error. The `SharedMemoryObject::getBase()` already IS a `ptr addrspace(3)` — pass it through directly.
**Instead:** Per-operand dispatch with `getSharedMemoryObjectFromStruct()` for memdesc operands (see Section 3.2).

### Anti-Pattern 2: Assuming `mlir::MemRefType`
**Why it's tempting:** AGENTS.md line 15 uses "memref" as loose terminology.
**Why it's wrong:** Triton's shared memory uses `ttg::MemDescType` (`TritonGPUTypes.td:23`), which is an entirely custom type with no relation to upstream MLIR's `MemRefType`. It lowers to `LLVM::LLVMPointerType` with address space 3, not to any memref LLVM type. Mixing APIs leads to impossible type conversions.
**Instead:** Use `ttg::MemDescType`, `SharedMemoryObject`, `getSharedMemoryObjectFromStruct()` — the established Triton infrastructure. All references documented in `.planning/research/STACK.md` Section 1.

### Anti-Pattern 3: Creating a New `gl.call_shared()` API
**Why it's tempting:** Separates concerns cleanly for implementation.
**Why it's wrong:** Fragments the user-facing API; creates duplicate inference, lowering, spec extraction, and linking code paths; forces users to know at call-site whether args include shared memory.
**Instead:** Extend `gl.call()` to accept both `tensor` and `shared_memory_descriptor` arguments. The semantic layer dispatches based on `isinstance(a, ...)`.

### Anti-Pattern 4: Double-Parsing the `.cu` File
**Why it's tempting:** The `_pre_compile_extern_calls()` method has both a suspended-compiler path (L771-806) and a fallback `compile_cuda_to_module` path (L808-832). It might seem simpler to re-parse the `.cu` in the fallback path for shared-memory args.
**Why it's wrong:** The existing v1.0 inference machinery already caches a suspended `CUDACompiler` per `.cu` file (via `_infer_hook._compilers` at L776). Re-parsing in the fallback path would cause a second clang parse of the same source — clang parse is the slowest part of the pipeline (~100-500ms per `.cu`).
**Instead:** The suspended-compiler path handles inference and compilation in a single parse. For shared-memory args, the suspended compiler must be passed `SharedTensorParameter` alongside `TensorParameter` in the request. The fallback path must also support `SharedTensorParameter`. **Verify:** The suspended compiler's `compileBitcode` route already goes through `_pre_compile_extern_calls` → the suspended path (L771-806). As long as `SharedTensorParameter` is in the `ParamTypes` variant and `CUDACompiler::BuildSharedTensor` exists, both paths work with a single parse.

### Anti-Pattern 5: Building a Parallel Clang AST Type System
**Why it's tempting:** `SharedTensor` seems different enough from `Tensor` to warrant separate builder/inspector classes.
**Why it's wrong:** The existing `TypeBuilder` and `TypeInspector` handle template type construction generically — the `Tensor` vs `SharedTensor` difference is just a different template name and parameter set. A parallel system duplicates the Shape construction, BasisGroup construction, Layout construction, and template instantiation logic.
**Instead:** Add `BuildSharedLinearLayout()` and `BuildSharedTensor()` methods to the existing `TypeBuilder`; add `ParseSharedLayoutType()` and `ParseSharedTensorType()` to the existing `TypeInspector`. Share the Shape template construction path (`buildShape()`) and BasisGroup template.

### Anti-Pattern 6: Using `getSharedMemoryBase()` Instead of `getSharedMemoryObjectFromStruct()` in Lowering
**Why it's tempting:** `getSharedMemoryBase()` returns a simple `ptr addrspace(3)`, and the extern call only needs the base pointer.
**Why it's wrong:** `getSharedMemoryBase()` reads `allocation.offset` directly from the `LocalAllocOp` — it returns the allocation's base without any subview offsets. If the `shared_memory_descriptor` was created via `memdesc_subslice`, `memdesc_trans`, `memdesc_reshape`, or `memdesc_reinterpret`, the offsets in `SharedMemoryObject.offsets` are not reflected in the base pointer. The callee would operate on the wrong memory location.
**Instead:** Always use `getSharedMemoryObjectFromStruct()` which reads the full LLVM struct (bases + accumulated offsets) produced by the subview op's lowering. The `SharedMemoryObject` correctly tracks all accumulated offsets.

---

## 6. Component Responsibility Matrix

| Component | File:Line | v1.1 Change | Risk |
|-----------|-----------|-------------|------|
| `gl.call()` frontend | `_core.py:775` | Pass-through memdesc args (no `to_tensor()` cast) | LOW — one `isinstance` branch |
| `call_extern()` validation | `_semantic.py:254` | Allow `shared_memory_descriptor` in args | LOW |
| `call_extern()` inference params | `_semantic.py:270-278` | Add `is_shared`, `offset_bases`, `block_bases`, `alignment` to arg_params | MED — inference hook must handle new keys |
| extern_call ODS | `TritonGPUOps.td:803` | `Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs` | **HIGH** — blast radius |
| `create_extern_call()` | `gluon_ir.cc:615-624` | Should work as-is after ODS change | LOW |
| `extractExternCallSpecs()` | `clang_compiler.cc:1169` | `isa<MemDescType>` branch | MED — new JSON fields |
| `SpecInput` struct | `clang_compiler.cc:1120-1142` | Add `isShared`, `offsetBases`, `blockBases`, `alignment` | LOW |
| JSON serialization | `clang_compiler.cc:1278` | New keys for shared inputs | LOW |
| `_pre_compile_extern_calls()` | `compiler.py:771-830` | Build `SharedTensorParameter` for shared inputs | MED — two parallel paths |
| `SharedLayoutInfo` | `clang_compiler.h` (NEW) | New struct | LOW |
| `SharedTensorParameter` | `clang_compiler.h` (NEW) | New struct | LOW |
| `CudaFuncRequest` | `clang_compiler.h:165` | Add variant | MED — changes all request consumers |
| `TypeBuilder` shared methods | `clang_compiler.cc` (NEW) | `BuildSharedLinearLayout`, `BuildSharedTensor` | MED — template construction |
| `TypeInspector` shared methods | `clang_compiler.cc` (NEW) | `ParseSharedLayoutType`, `ParseSharedTensorType` | MED — template parsing |
| `DispatchTypeParsing` | `clang_compiler.h:278` | Branch for `SharedTensor` | LOW |
| Python bindings | `llvm.cc` (NEW) | `SharedTensorParameter` pybind11 class | LOW |
| SharedMemoryObject | `Utility.h:374-448` | UNCHANGED (reused) | NONE |
| `getSharedMemoryObjectFromStruct` | `Utility.cpp:1428-1452` | UNCHANGED (reused) | NONE |
| `getSharedMemoryBase` | `Utility.cpp:1569-1589` | UNCHANGED | NONE |
| ExternCallOpConversion | `ExternCallOpToLLVM.cpp:141-152` | Per-operand dispatch (shared vs distributed) | **HIGH** — LLVM type correctness |
| `promoteOperands` | `ExternCallOpToLLVM.cpp:137` | UNCHANGED (handles both types) | NONE |
| `linkBitcodeToModule` | `clang_compiler.cc` | UNCHANGED | NONE |
| `SharedLinearLayout` template | `tt_plugin.cu` (NEW area) | New C++ template | MED — design review needed |
| `SharedTensor` template | `tt_plugin.cu` (NEW area) | New C++ template | MED — reuse vs. new decision |

---

## 7. Key Integration Verification Points

| Test Point | What It Validates | How To Test |
|------------|------------------|-------------|
| ODS verification | `extern_call` accepts both tensor and memdesc operands | `triton-opt` on MLIR with mixed inputs |
| Spec extraction | `extractExternCallSpecs` doesn't crash on MemDescType | Unit test / lit test with memdesc operand |
| JSON backward compat | Tensor-only specs unchanged; shared specs add optional fields | Parse both JSON variants in Python |
| Shared clang AST | `BuildSharedTensor` produces valid `SharedTensor<T,Shape,SL>&` | clang AST dump after construction |
| TypeInspector round-trip | Parse what BuildSharedTensor constructs | Equality check: param == ParseSharedTensorType(BuildSharedTensor(param)) |
| Function resolution | `LookupFunction` matches device fn with mixed args | Resolve `my_fn(Tensor&, SharedTensor&)` with concrete types |
| Lowering type match | Call-site `ptr addrspace(3)` matches callee `ptr addrspace(3)` | Dump LLVM IR after lowering, check arg types |
| E2E shared R/W | Callee reads and writes shared memory through `SharedTensor&` | Pytest: allocate shared → call `gl.call()` → load shared → compare |
| Regression | Existing tensor-only extern calls unchanged | All 6 existing `test_extern_call.py` tests pass |

---

## Sources

- `TritonGPUOps.td:786-814` — ExternCallOp ODS (`Variadic<TT_Tensor>:$inputs`)
- `TritonGPUTypes.td:23-84` — MemDescType definition
- `TritonGPUAttrDefs.td:395-426` — SharedLinearEncodingAttr
- `TritonGPUAttrDefs.td:1496-1501` — SharedMemorySpaceAttr
- `LinearLayoutConversions.cpp:1376-1392` — `toLinearLayout(MemDescType)`
- `clang_compiler.cc:1152-1219` — `extractExternCallSpecs()`
- `clang_compiler.cc:730-768` — `CUDACompiler::BuildTensor()` (pattern for new `BuildSharedTensor`)
- `clang_compiler.h:129-141` — `LayoutInfo`, `TensorParameter`
- `clang_compiler.h:165-169` — `CudaFuncRequest`
- `clang_compiler.h:224-280` — TypeBuilder, TypeInspector
- `clang_compiler.h:286-296` — FunctionResolver
- `compiler.py:709-807` — `_pre_compile_extern_calls()`
- `compiler.py:771-806` — Suspended-compiler path (inference reuse)
- `compiler.py:808-832` — Fallback `compile_cuda_to_module` path
- `ExternCallOpToLLVM.cpp:111-293` — ExternCallOpConversion
- `Utility.h:374-448` — SharedMemoryObject
- `Utility.h:454-457` — `getSharedMemoryObjectFromStruct()`
- `Utility.cpp:1428-1452` — `getSharedMemoryObjectFromStruct()` implementation
- `Utility.cpp:1569-1589` — `getSharedMemoryBase()`
- `TritonNVIDIAGPUToLLVM/TargetInfo.cpp:620` — `getSharedAddressSpace() returns 3`
- `gluon_ir.cc:308-318` — `get_shared_mem_desc_ty`
- `gluon_ir.cc:454-466` — `get_shared_linear_layout`
- `gluon_ir.cc:615-624` — `create_extern_call`
- `gluon_ir.cc:235-242` — `layoutToGluon` SharedLinearLayout branch
- `_core.py:185-243` — `shared_memory_descriptor_type`
- `_core.py:291-339` — `shared_memory_descriptor`
- `_core.py:775-809` — `gl.call()`
- `_semantic.py:250-319` — `call_extern()`
- `_layouts.py:631-673` — `SharedLinearLayout`
- `tt_plugin.cu:1-185` — Existing C++ templates (Shape, TensorLayout, Tensor)
- `.planning/research/STACK.md` — Companion research on exact types/APIs (authoritative for MLIR type details)
