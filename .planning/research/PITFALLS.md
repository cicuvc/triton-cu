# Domain Pitfalls: Shared Memory Interop for `gl.call()` (v1.1)

**Domain:** Compiler internals — CUDA/MLIR/Clang shared-memory interop
**Researched:** 2026-07-12
**Overall confidence:** HIGH (every pitfall grounded in `file:line` in this repo)

---

## Executive Summary

Adding shared-memory (`addrspace 3`) argument passing to `gl.call()` introduces correctness risks at four distinct compiler layers: MLIR op signature, Clang AST type construction, LLVM lowering, and CUDA device-function linkage. The highest-risk pitfall is **address-space mismatch** — the existing tensor-arg lowering path uses `alloca+store+ptr` which produces `ptr addrspace(0)`, but shared-memory callees expect `ptr addrspace(3)`. Passing the wrong address space silently corrupts data. The second-highest is **swizzle/offset-bases mismatch** between the MLIR `SharedLinearEncodingAttr` and the CUDA C++ `SharedLinearLayout` template — both must compute identical byte offsets from logical tensor indices. Several of these pitfalls are guarded by the v1.0 `parse-counter` infrastructure (`sExternCudaParseCount` at `clang_compiler.cc:57`, asserted at `compiler.py:683`), which must be extended — not broken — by the v1.1 additions.

---

## Critical Pitfalls (Rewrite-Class)

Mistakes that cause silent data corruption, hard crashes, or require architectural rework.

### Pitfall 1: Address-Space Mismatch in Callee Argument Lowering

**What goes wrong:** The existing lowering path in `ExternCallOpToLLVM.cpp:143-152` applies the `alloca + store + ptr` pattern to EVERY operand — but this produces `ptr addrspace(0)`. For shared-memory operands, the callee expects `ptr addrspace(3)`. If an addrspace-0 pointer is passed where the callee expects addrspace 3, the NVPTX backend either (a) silently emits loads/stores to address space 0 (generic memory), producing wrong data, or (b) if the callee's LLVM IR explicitly uses `ptr addrspace(3)`, the bitcast between address spaces is rejected by the verifier.

**Why it happens:** `ExternCallOpToLLVM.cpp:143-152` does not check operand type:
```cpp
// ExternCallOpToLLVM.cpp:143-152 — CURRENT (attacks every operand)
auto ptrTy = LLVM::LLVMPointerType::get(ctx, 0);  // addrspace 0!
for (unsigned i = 0; i < numTensorArgs; ++i) {
  auto structVal = promotedOperands[i];
  Value stackPtr = LLVM::AllocaOp::create(..., ptrTy, structTy, one, 0);
  b.store(structVal, stackPtr);  // stores to addrspace 0
  promotedOperands[i] = stackPtr;  // passes addrspace 0 ptr
}
```
The callee function, compiled by clang from a `__device__` function with a `SharedTensor<...>&` parameter, naturally produces `ptr addrspace(3)` in its LLVM IR signature — because clang's NVPTX target puts `__device__`-function arguments that are references into the appropriate address space. The types mismatch.

**Consequences:** Silent data corruption (reads/writes to wrong memory address space). No compiler error — the LLVM `call` instruction may accept the mismatched pointer types depending on verification strictness, but the PTX `ld.shared` / `st.shared` instructions will access the wrong memory.

**Prevention:**

1. **In the lowering** (`ExternCallOpToLLVM.cpp`): For memdesc operands, extract `SharedMemoryObject` and pass `getBase()` directly — do NOT alloca/store:
   ```cpp
   if (isa<ttg::MemDescType>(operandType)) {
     auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
         loc, promotedOperands[i], elemTy, rewriter);
     promotedOperands[i] = smemObj.getBase();  // ptr addrspace(3)
   } else {
     // existing alloca+store path for tensor args
   }
   ```
   The `SharedMemoryObject::getBase()` at `Utility.h:397-402` returns the addrspace-3 base pointer. The `getSharedMemoryObjectFromStruct()` at `Utility.cpp:1428-1452` correctly unpacks the LLVM struct — bases first (pointer types), then offsets (i32 types).

2. **In the callee LLVM IR** (clang side): No changes needed. Clang's CodeGen for `__device__` functions with `--cuda-device-only` naturally assigns `ptr addrspace(3)` to reference parameters. This is confirmed by the NVPTX target's address space conventions (LLVM 23.0.0git, stable since LLVM 15).

**Detection (warning signs):**
- **LLVM IR inspection:** `grep 'addrspace'` on the callee function — the shared-memory parameter should show `ptr addrspace(3)`, not `ptr`. The caller's `call` instruction must pass a matching `ptr addrspace(3)` value.
- **Runtime symptom:** Reads from shared memory return garbage (uninitialized memory or global memory values). Writes to shared memory silently land in the wrong address space — data written but never visible to other threads.
- **PTX inspection:** The PTX for the callee should contain `ld.shared` / `st.shared` instructions. If the pointer was passed with wrong addrspace, PTXAS may emit `ld.global` instead.
- **Verification:** After `linkBitcodeToModule()` in `compiler.py:673`, `verifyModule` may reject `call` instructions with address-space-mismatched arguments, producing a `RuntimeError("LLVM module verification failed after extern linking")`.

**Preventing phase:** Phase 2 (MLIR Lowering) or Phase 3 (Verification). Must be fixed before any E2E test can pass.

---

### Pitfall 2: `cast<RankedTensorType>` Crash on MemDesc Operands

**What goes wrong:** `extractExternCallSpecs()` at `clang_compiler.cc:1170` unconditionally casts every operand to `RankedTensorType`:
```cpp
// clang_compiler.cc:1169-1170
for (auto operand : op.getInputs()) {
    auto tensorTy = cast<RankedTensorType>(operand.getType());  // CRASHES on MemDescType!
```
When a `ttg.extern_call` carries a `MemDescType` operand (shared memory descriptor), the `cast<>` triggers an LLVM assertion failure in debug builds and undefined behavior in release builds. This happens BEFORE the CUDA compiler is even invoked — the spec extraction fails during `_pre_compile_extern_calls()`.

**Why it happens:** The spec extraction was built under the assumption that all `extern_call` inputs are distributed tensors (which was true for v1.0). `MemDescType` is a different MLIR type class and cannot be cast to `RankedTensorType`.

**Consequences:** Hard crash. No externally-visible error message — the `cast<>` fails with an `llvm_unreachable` internally. In release builds, the program may read invalid memory and crash later.

**Prevention:** Split operand handling in `extractExternCallSpecs()`:
```cpp
for (auto operand : op.getInputs()) {
    auto operandType = operand.getType();
    if (auto tensorTy = dyn_cast<RankedTensorType>(operandType)) {
        // existing tensor path: extract reg/lane/warp bases
    } else if (auto memDescTy = dyn_cast<ttg::MemDescType>(operandType)) {
        // NEW: shared-memory path
        auto ll = mlir::triton::gpu::toLinearLayout(memDescTy);  // LinearLayoutConversions.cpp:1376
        SpecInput input;
        input.is_shared = true;
        input.shape = ...; // from memDescTy.getShape()
        input.dtype = ...; // from memDescTy.getElementType()
        input.offset_bases = ll.getBases().lookup("offset");  // or appropriate key
        input.block_bases = ll.getBases().lookup("block");
        input.alignment = cast<ttg::SharedLinearEncodingAttr>(
            memDescTy.getEncoding()).getAlignment();  // Dialect.cpp:2061
        // ...
    }
}
```

Note: The `toLinearLayout` for `MemDescType` at `LinearLayoutConversions.cpp:1376` uses `getAllocShape()` to handle subviews. The shared encoding's `toLinearLayout()` at `Dialect.cpp:2061` returns the `LinearLayout` directly without broadcasting — the shape must match.

**Detection:** Any E2E test with a `gl.call()` that includes a `shared_memory_descriptor` argument will crash here.

**Preventing phase:** Phase 1 (Op Signature & Spec Extraction). Must be fixed before CUDA compilation can be tested.

---

### Pitfall 3: ODS `Variadic<TT_Tensor>` Rejects MemDescType

**What goes wrong:** The `ttg::ExternCallOp` is defined with `Variadic<TT_Tensor>:$inputs`:
```tablegen
// TritonGPUOps.td:802-808
let arguments = (ins
    Variadic<TT_Tensor>:$inputs,
    ...
);
```
`MemDescType` does NOT satisfy the `TT_Tensor` constraint. Attempting to build the op with a memdesc value will be rejected by the MLIR verifier before the op is even created.

**Why it happens:** The ODS constraint was written when `extern_call` only accepted tensor operands. Shared memory was not in scope.

**Consequences:** MLIR verification failure at op creation time. The `create_extern_call` at `gluon_ir.cc:620` will fail because `self.create<ttg::ExternCallOp>(resultTypes, args, ...)` verifies the op immediately.

**Prevention:** Relax the type constraint using MLIR's `AnyTypeOf` or add a separate variadic for memdesc inputs. Options in order of preference:

**Option A (AnyTypeOf — clean):**
```tablegen
let arguments = (ins
    Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>:$inputs,
    ...
);
```
**Option B (separate variadic — preserves existing TT_Tensor path):**
```tablegen
let arguments = (ins
    Variadic<TT_Tensor>:$tensor_inputs,
    Optional<Variadic<TTG_MemDescType>>:$shared_inputs,
    ...
);
```
Option B is safer because it prevents existing tensor-typed code from breaking inadvertently. However, it requires position-management logic so that the combined `[tensors..., memdescs...]` ordering is preserved for the callee signature.

**Detection:** Before any lowering work begins, creating the MLIR op with a memdesc handle will fail with a verifier error message.

**Preventing phase:** Phase 1 (Op Signature & Spec Extraction). Must be the very first code change.

---

### Pitfall 4: Swizzle/Offset-Bases Mismatch Between MLIR and CUDA C++

**What goes wrong:** `SharedLinearLayout` (Python at `_layouts.py:631`, mapped to `SharedLinearEncodingAttr` in MLIR at `TritonGPUAttrDefs.td:395`) defines a byte-offset computation from logical tensor indices using `offset_bases` + `block_bases` + `alignment`. The CUDA C++ `SharedLinearLayout` template (new in `tt_plugin.cu`) must compute the **identical** byte offset. A mismatch means the callee reads/writes the wrong shared-memory addresses — silently wrong results.

**Why it happens:** The MLIR side computes offsets via `LinearLayout` composition:
```
byte_offset = sum_i( logical_index[i] * offset_base[i] ) + sum_j( block_id[j] * block_base[j] )
```
The `LinearLayout` infrastructure at `Dialect.cpp:2061` stores basis vectors keyed by `StringAttr("offset")` and `StringAttr("block")`. The CUDA C++ side must implement the same math manually — there is no shared library, no cross-compilation check, and no runtime validation that the two sides agree.

The `SharedLinearLayout` shape computation at `_layouts.py:660-666` derives logical dimensions from `max_stride` by computing `1 << s.bit_length()` for each dimension — this algorithm must match the C++ side's dimension inference from the same bases.

**Consequences:** Wrong byte offsets → wrong shared memory addresses → silent data corruption. The most insidious case: a single basis vector off-by-one produces partially correct results for small tensors that happen to fit in the correct address range, but wrong results for larger tensors.

**Prevention:**

1. **Make the address-computation math a single source of truth.** The safest approach is to have the C++ `SharedLinearLayout` template compute offsets in a `__device__` helper that produces the same value the `LinearLayout` composition would. The `tt_plugin.cu` template should:
   ```cpp
   template<typename TShape, typename TSharedLayout>
   __device__ size_t shared_offset(const TShape& idx, const TSharedLayout& layout) {
       size_t off = 0;
       for (int b = 0; b < TSharedLayout::NUM_OFFSET_BASES; ++b) {
           for (int d = 0; d < TShape::RANK; ++d)
               off += idx[d] * TSharedLayout::offset_bases[b][d];
       }
       for (int b = 0; b < TSharedLayout::NUM_BLOCK_BASES; ++b) {
           for (int d = 0; d < TShape::RANK; ++d)
               off += blockIdx[d] * TSharedLayout::block_bases[b][d];
       }
       return off;
   }
   ```
   This formula must exactly match the `LinearLayout` composition: `LinearLayout({offsetBases, blockBases}, outDims)` at `gluon_ir.cc:102-103`.

2. **Alignment check:** Both sides must agree on alignment. The MLIR `SharedLinearEncodingAttr` stores `layoutAlignment` as a `"unsigned"` parameter (`TritonGPUAttrDefs.td:421`). The C++ side must carry the same alignment value. The byte offset must satisfy `(byte_offset % alignment == 0)` for the base pointer.

3. **Testing:** The most effective verification is an E2E test that writes known values through the `SharedTensor&` at specific logical positions, then reads them back through `shared_memory_descriptor.load()` with a known-good distributed layout, and verifies the values match. Each basis should be exercised independently.

**Detection:** The only reliable detection is E2E testing. MLIR-side `toLinearLayout` can be exported and compared to C++ `__device__` path tracing for the same bases.

**Preventing phase:** Phase 3 (Verification) must contain a dedicated swizzle-correctness test. Phase 2 (Lowering) must include a debug-mode assertion or comment documenting the offset formula at the point where bases flow from MLIR to clang AST.

---

### Pitfall 5: Subview/Offset Accumulation Not Preserved

**What goes wrong:** When a user calls `smem.slice(start, length, dim)` (`_core.py:446`) or `smem.index(index)` (`_core.py:464`) before passing the result to `gl.call()`, the memdesc's `SharedMemoryObject.offsets` have been modified by `ttg.memdesc_subslice` or `ttg.memdesc_index`. The `getSharedMemoryObjectFromStruct()` at `Utility.cpp:1428-1452` correctly unpacks both bases AND offsets from the LLVM struct, but the lowered code must apply `getShmemOffset()` to compute the actual ptr passed to the callee. Passing `getBase()` without applying the accumulated offsets passes the wrong starting address to the callee.

**Why it happens:** `SharedMemoryObject` stores bases separately from offsets (`Utility.h:442-448`). The base pointer (`getBase()`) points to the START of the allocation (addrspace-3 GEP from stack pointer via `getSharedMemoryBase()` at `Utility.cpp:1569-1589`). The `offsets` vector accumulates affine offsets from subview operations. In the extern-call lowering, the callee expects the base+offset pointer — not just the base.

**Consequences:** The callee sees the allocation's base address, not the subview's start. If slice start = 32 elements and each element is 4 bytes, the callee reads/writes 128 bytes before the intended subview — corrupting adjacent data.

**Prevention:** In the lowering, for memdesc operands, compute the full offset from `getShmemOffset()` (or manually add `smemObj.getOffsets()` to the base pointer via GEP):
```cpp
auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, llvmStruct, elemTy, rewriter);
Value base = smemObj.getBase();  // ptr addrspace(3), base of allocation
Value offset = smemObj.getShmemOffset(loc, rewriter, memDescTy);  // accumulated offsets
// GEP: base + offset
auto sharedPtrTy = base.getType();  // ptr addrspace(3)
auto i8Ty = IntegerType::get(ctx, 8);
base = b.gep(sharedPtrTy, i8Ty, base, offset);
promotedOperands[i] = base;
```
The `getShmemOffset()` method at `Utility.h:430-431` computes the accumulated byte offset accounting for padding and subviews. This is the correct offset to add to the base pointer.

**Detection:** E2E test: allocate shared memory, `slice()` a subview, pass to `gl.call()`, verify the callee writes to the correct subview addresses (read back and check the values don't land in adjacent slices).

**Preventing phase:** Phase 2 (MLIR Lowering). Must be handled alongside the address-space fix (Pitfall 1).

---

### Pitfall 6: Alignment Violation in Vectorized Shared Access

**What goes wrong:** `SharedLinearLayout.alignment` (`_layouts.py:636`, default 16) enforces a minimum alignment for shared memory access. If the base pointer GEP or offset computation produces an address that is not a multiple of `alignment`, vectorized shared memory access instructions (e.g., PTX `ld.shared.v4.f32`) will fault with a misaligned address error on the GPU.

**Why it happens:** The callee's CUDA C++ code may issue vectorized loads/stores through the `SharedTensor&` reference. If the `SharedLinearLayout` specifies `alignment: 16` but the passed pointer is only 4-byte aligned (e.g., because a `memdesc_slice` produced an odd starting offset, or because the `getSharedMemoryBase()` GEP computed an offset not divisible by the element size × alignment), the GPU hardware raises a misaligned-address fault.

**Consequences:** GPU fault (misaligned address exception). The kernel crashes. PTXAS may warn but won't fail — this is a runtime hardware fault.

**Prevention:**
1. In the lowering, after computing the full pointer (base + offsets from subview), verify `(ptr_address % alignment == 0)` at compile time where possible (constant offsets). At runtime, add a debug-mode assertion.
2. The `SharedLinearLayout` alignment should propagate to both sides. On the MLIR side, `SharedLinearEncodingAttr::getAlignment()` at `Dialect.cpp` returns `static_cast<int32_t>(getLayoutAlignment())`. On the C++ side, the `SharedLinearLayout` template carries the same alignment value.
3. When `toLinearLayout` is called on the `MemDescType` at `LinearLayoutConversions.cpp:1376`, the encoding's alignment is retrieved — this should be included in the `SpecInput` JSON and plumbed through to the C++ template construction.

**Detection:** E2E test with a non-trivial `SharedLinearLayout` (full swizzle with odd bases that produce non-16-aligned offsets for some indices). Verify the callee can read/write all index positions without faulting.

**Preventing phase:** Phase 3 (Verification). Phase 2 should propagate alignment correctly; Phase 3 verifies.

---

### Pitfall 7: Mixed Distributed+Shared Argument Ordering Mismatch in Callee Signature

**What goes wrong:** The `ttg.extern_call` operands are positionally ordered: `[tensor_0, tensor_1, ..., memdesc_0, memdesc_1, ...]`. The lowering processes them in order, injecting addrspace-0 `alloca` pointers for tensors and addrspace-3 `getBase()` for shared memory. The callee function signature (compiled by clang from the user's `.cu`) must have parameters in the EXACT SAME order. If the MLIR operands are `[tensor, memdesc]` but the C++ function signature is `(SharedTensor&, const Tensor&)`, the callee's first parameter expects `ptr addrspace(3)` but receives `ptr addrspace(0)` — an address-space mismatch.

**Why it happens:** The callee function is compiled independently from the MLIR module. The spec extraction (`extractExternCallSpecs()`) preserves operand order from the MLIR walk, and `_pre_compile_extern_calls()` builds `CudaFuncRequest.paramTypes` in the same order. However, the user's `.cu` file may define the function with parameters in any order. Overload resolution (`FunctionResolver::LookupFunction`) matches by type, not by position relative to the MLIR operand order. The lowered call builds its argument list from the MLIR operand order — NOT from the C++ function's declared parameter order.

**Consequences:** Address-space mismatch on specific parameters — not the first shared-memory argument, but the WRONG shared-memory argument or a tensor parameter that should have been a shared-memory parameter. The LLVM verifier may catch this (different address spaces on `call` vs `callee`), but the NVPTX backend may or may not reject it.

**Prevention:** The order of `CudaFuncRequest.paramTypes` must match the C++ function's declared parameter order — NOT the MLIR operand order. The `FunctionResolver::LookupFunction()` already deduces the parameter types from the resolved `FunctionDecl`. The lowering must reorder the MLIR operands to match the C++ function parameter order, or alternatively, the MLIR operands must be emitted in C++ parameter order from the start.

The safest approach: at MLIR creation time (`_semantic.py`), order the operands to match the C++ function's declared parameter order. This requires the inference hook to return parameter type information alongside return type information — a new field in the inference result. Alternatively, store the parameter ordering as a module attribute so the lowering can permute operands.

**Detection:** Inspection of the lowered LLVM IR call: the `call @fn(ptr addrspace(0) %1, ptr addrspace(3) %2)` must match the callee's `define void @fn(ptr addrspace(0) %arg0, ptr addrspace(3) %arg1)` — same addrspaces in the same positions.

**Preventing phase:** Phase 2 (Lowering & Integration). Requires coordination between spec extraction, CudaFuncRequest construction, and lowering operand permutation.

---

### Pitfall 8: Double-Parse of .cu Breaking the Single-Parse Guard

**What goes wrong:** The v1.0 infrastructure has a per-compile parse counter (`sExternCudaParseCount` at `clang_compiler.cc:57`, exposed via `get_extern_cuda_parse_count()` at `llvm.cc:1021`) that is asserted at `compiler.py:683-686`:
```python
assert parse_count_delta == distinct_cu, (
    f"extern CUDA parse count mismatch: ...")
```
The assertion fails if the `.cu` file is parsed more than once per compile — e.g., once at semantic time for return-type inference and again at llir time for bitcode compilation. The suspended-compiler pattern (hook at `compiler.py:190-218`, reuse at line 776) prevents this for v1.0. For v1.1, the shared-memory type construction (`BuildSharedLinearLayout`, `BuildSharedTensor`) must use the already-suspended compiler — NOT trigger a new parse.

**Why it happens:** Each `CUDACompiler` construction (`clang_compiler.cc:588-693`) increments `sExternCudaParseCount` when `parse()` is called. The suspended compiler (`InferExternCallResult.create_and_suspend` at `compiler.py:203-218`) parses once and parks the AST context. If the v1.1 shared-memory spec extraction path creates a new `CUDACompiler` instead of reusing the suspended one, the parse counter increments and the assertion fires.

**Consequences:** The `assert` at `compiler.py:683` fires → hard crash. Even if the assertion is removed, a double parse wastes CPU and may produce different bitcode (different LLVM module instances) that can't be linked.

**Prevention:** Shared-memory `TensorParameter` variants must flow through the EXISTING suspended-compiler path. The `CudaFuncRequest.paramTypes` variant at `clang_compiler.h:167` should gain a `SharedTensorParameter` variant:
```cpp
// clang_compiler.h:165-169 — EXTEND:
struct CudaFuncRequest {
    std::string Symbol;
    std::vector<std::variant<ScalarType, TensorParameter, SharedTensorParameter>> ParamTypes;
    bool UseFastMath = false;
};
```
The `compile_bitcode` path at `compiler.py:776-806` needs to build `SharedTensorParameter` from shared-memory `SpecInput` entries and pass them to the suspended compiler's `CudaFuncRequest`.

**Detection:** Any change that causes `get_extern_cuda_parse_count()` to increment by more than the number of distinct `.cu` files during a single compile triggers the assertion. The delta should equal `distinct_cu` (from `metadata["extern_distinct_cu"]`).

**Preventing phase:** Phase 1 (Hook Integration). The `InferExternCallResult` hook and `CudaFuncRequest` must be extended BEFORE phase 2 begins.

---

## Moderate Pitfalls

### Pitfall 9: MLIR Verification Failure After Op Signature Relaxation

**What goes wrong:** After relaxing the `extern_call` op to accept `MemDescType` (Pitfall 3), downstream passes that read the op's result types — e.g., `convert_layout` inserted by `patch_extern_call_result_types` at `clang_compiler.cc:950` — may encounter type inconsistencies. The result types are still `TensorType`s built from `inferred_dtype` + `result_shape` + `result_layout`, and they don't depend on shared-memory operands. However, if the relaxed ODS constraint inadvertently allows mixing MemDescType results, or if the `SameVariadicOperandSize` trait produces false validation, the module may fail verification.

**Why it happens:** MLIR's verifier runs after every op insertion. The op's `SameVariadicOperandSize` trait at `TritonGPUOps.td:787` constrains operand vs. result variadic sizes. If inputs become mixed-type but results remain TT_Tensor-only, the trait may need adjustment.

**Prevention:** Use `AnyTypeOf<[TT_Tensor, TTG_MemDescType]>` only on the INPUTS variadic. Results stay `Variadic<TT_Tensor>`. Remove or adjust `SameVariadicOperandSize` if it conflicts with mixed input types. Test with `ModuleOp::verify()` after building the op.

**Detection:** MLIR verification errors at op creation or module verification time. The `create_extern_call` at `gluon_ir.cc:620` will throw a Python exception if verification fails.

**Preventing phase:** Phase 1 (Op Signature & Spec Extraction).

---

### Pitfall 10: `shared_memory_descriptor` Validation in `call_extern()` Rejects MemDesc Args

**What goes wrong:** `_semantic.py:254` validates all args with `isinstance(a, ttgl.tensor)`. A `shared_memory_descriptor` object fails this check, raising a Python error before the MLIR op is ever built.

**Why it happens:** The arg validation was written for v1.0 when only distributed tensors could be arguments.

**Consequences:** User-facing `RuntimeError`: "all arguments must be tensors but got shared_memory_descriptor". The `gl.call()` invocation fails immediately.

**Prevention:** Extend the check to also accept `shared_memory_descriptor`:
```python
# _semantic.py:253-255
for a in args:
    _check(isinstance(a, (ttgl.tensor, ttgl.shared_memory_descriptor)),
           lambda: f"all arguments must be tensors or shared memory descriptors but got {type(a)!r}")
```
Also, the type-parameter construction (`arg_params` at _semantic.py:270-276) needs to encode whether an arg is shared memory so the inference hook can build `SharedTensorParameter` instead of `TensorParameter`.

**Prevention phase:** Phase 1 (Frontend Integration).

---

## Minor Pitfalls

### Pitfall 11: Callee Ret-Type Fix Code Corrupted by Shared-Memory Args

**What goes wrong:** The `linkBitcodeToModule` ret-type fix at `clang_compiler.cc:1340-1376` iterates over `ReturnInst`s to launder named `%struct.Tensor` into literal struct types. If the callee function has NO return value (void-returning shared-memory writer), this code path is fine. But if the callee returns a distributed tensor AND takes a `SharedTensor&` arg, the new parameter types may introduce clang-generated metadata or struct types that the ret-type fix doesn't expect — causing bitcode corruption.

**Why it happens:** The fix assumes that return-type mismatches involve `Tensor<T, Shape, Layout>` struct types only. A new `SharedTensor` struct type in the bitcode may introduce a struct type the Fix doesn't know about but still encounters during the iteration, potentially mismatching.

**Prevention:** This is most likely a non-issue because the return type is still `Tensor<...>` (unchanged from v1.0) — `SharedTensor` is argument-only. However, the `StripDebugInfo` at `clang_compiler.cc:1395` and `DISubprogram` stripping at line 1378-1390 may be affected by the new types. Verify with a void-returning extern call that takes `SharedTensor&` (the simplest case).

**Detection:** If `linkBitcodeToModule` corrupts bitcode, `verifyModule` at `compiler.py:673` will fail with a hard-to-diagnose error about struct type mismatches.

**Preventing phase:** Phase 3 (Verification). Test early with void-returning callees.

---

## Read+Write Hazard: Missing `__syncthreads()` Semantics

### Pitfall 12: Write-Back Visibility Without Synchronization

**What goes wrong:** `gl.call()` with a `SharedTensor<...>&` parameter allows the CUDA device function to WRITE to shared memory. After the call returns, GLUON CODE (on the same or different threads) may read that shared memory. However, `gl.call()` does NOT insert a `__syncthreads()` (CTA-wide barrier) before or after the call. If the caller's threads reach different execution points — some threads still inside `gl.call()`, others already reading the written shared memory — the writes are not guaranteed to be visible. This is a **race condition** at the warp/CTA level.

**Why it happens:** `gl.call()` emits a single LLVM `call` instruction per thread. The callee is inlined via `alwaysinline` + O3 (`clang_compiler.cc:1392`). There is no `llvm.nvvm.barrier0` or equivalent inserted. The Triton/Gluon `gl.barrier()` function at `_core.py:703` is available but must be called explicitly by the user. There's no annotation or warning that `gl.call()` with shared-memory write-back needs an explicit barrier.

**Consequences:** Intermittent data corruption — depends on warp scheduling, CTA size, and the GPU architecture's memory consistency model. Hardest to debug because it's non-deterministic.

**Prevention:**
1. **Documentation:** `gl.call()` docstring must explicitly warn about the absence of barrier semantics. The user is responsible for inserting `gl.barrier()` around shared-memory-accessing `gl.call()` invocations.
2. **Option (not for v1.1):** Consider adding an optional `sync_before`/`sync_after` parameter to `gl.call()` in a future milestone.
3. **Compiler hint:** The lowering (Phase 2) can emit a debug-mode assertion or comment noting the barrier absence.

**Detection:** Tests with multiple warps concurrently writing/reading shared memory through `gl.call()` may pass ~80% of the time and fail ~20% due to race conditions. The only reliable detection is stress testing with large CTA sizes and `CUDA_LAUNCH_BLOCKING=1`.

**Preventing phase:** Not a compile-time fix — this is a **documentation and testing** concern. Phase 3 (Verification) must include a test that documents the correct barrier usage pattern. Phase 4 (Documentation/Integration) must include the user-facing warning.

---

## Pitfall-to-Phase Mapping

| Pitfall | # | Phase Should Prevent It | Phase Name |
|---------|---|------------------------|------------|
| Address-space mismatch | 1 | Phase 2 | MLIR Lowering (ExternCallOpToLLVM) |
| `cast<RankedTensorType>` crash | 2 | Phase 1 | Op Signature & Spec Extraction |
| ODS `Variadic<TT_Tensor>` reject | 3 | Phase 1 | Op Signature & Spec Extraction |
| Swizzle/offset-bases mismatch | 4 | Phase 3 | Verification (E2E swizzle test) |
| Subview offset accumulation | 5 | Phase 2 | MLIR Lowering (alongside Pitfall 1) |
| Alignment violation | 6 | Phase 3 | Verification (edge-case test) |
| Mixed arg ordering | 7 | Phase 2 | Lowering & Integration |
| Double-parse | 8 | Phase 1 | Hook Integration (CudaFuncRequest extension) |
| MLIR verification post-relaxation | 9 | Phase 1 | Op Signature & Spec Extraction |
| `call_extern()` validation reject | 10 | Phase 1 | Frontend Integration |
| Ret-type fix corruption | 11 | Phase 3 | Verification |
| Read-write hazard (barrier) | 12 | Phase 3 + 4 | Verification + Documentation |

---

## "Looks Done But Isn't" Checklist

These are the verification checks that distinguish "code compiles" from "actually works correctly":

### Address Space (Pitfall 1)
- [ ] **Inspect LLVM IR** for the caller: `grep 'ptr addrspace'` on the callee side → shared-memory args show `ptr addrspace(3)`.
- [ ] **Inspect LLVM IR** for the callee: the `define` line for the device function has `ptr addrspace(3)` at the correct parameter position.
- [ ] **PTX inspection:** The callee PTX contains `ld.shared` / `st.shared` (not `ld.global` / `st.global`) for shared-memory accesses.
- [ ] **E2E test:** Write known values through the `SharedTensor&` path, read back through `shared_memory_descriptor.load()`, verify bit-for-bit match (not "close enough").

### Swizzle Correctness (Pitfall 4)
- [ ] **Single-basis test:** A `SharedLinearLayout` with ONE offset basis and ONE block basis → callee reads/writes the correct element.
- [ ] **Multi-basis test:** Bases with non-trivial pattern (full swizzle) → callee computes the correct byte offset for every element.
- [ ] **Alignment test:** Callee accesses with `alignment: 64` (non-default) → all accesses satisfy alignment, no GPU faults.
- [ ] **Shape inference:** The C++ `SharedLinearLayout` derives the same logical shape as `_layouts.py:660-666` (the `max_stride` → `bit_length` algorithm).

### Subview Offset (Pitfall 5)
- [ ] **Slice test:** Allocate `[M, N]` shared memory, slice `[:M/2, :]`, pass to `gl.call()`, verify no writes beyond the subview boundaries.
- [ ] **Index test:** Allocate multi-dim shared memory, `index(offset)`, pass to `gl.call()`, verify the callee sees the correct subview start.
- [ ] **Chained subview test:** `smem.slice().permute().index()` → pass to `gl.call()`, verify accumulated offsets are correct.

### Mixed Arg Order (Pitfall 7)
- [ ] **Single shared arg:** `gl.call(fn, [tensor, smem], ...)` — callee signature is `fn(const Tensor&, SharedTensor&)`.
- [ ] **Reversed order:** `gl.call(fn, [smem, tensor], ...)` — callee signature is `fn(SharedTensor&, const Tensor&)`.
- [ ] **Multiple shared args:** `gl.call(fn, [tensor, smem1, smem2], ...)` — callee sees correct shared memory at each position.

### Single-Parse Integrity (Pitfall 8)
- [ ] **Parse count assertion:** `compiler.py:683-686` passes with shared-memory args — `parse_count_delta == distinct_cu`.
- [ ] **Compiler reuse:** The suspended compiler path at `compiler.py:776` is used (not the fallback `compile_cuda_to_module` at line 808).
- [ ] **Multiple calls, same .cu:** Two `gl.call()` invocations referencing different functions in the same `.cu` share one suspended compiler.

### Write-Back Safety (Pitfall 12)
- [ ] **Documentation check:** `gl.call()` API docs mention that `gl.barrier()` is needed around shared-memory-accessing calls.
- [ ] **Test with barrier:** E2E test with `gl.barrier()` before and after write-back `gl.call()` — results are deterministic.
- [ ] **Stress test:** Multiple warps, large CTA, many iterations — no data races (passes 100/100 runs with `CUDA_LAUNCH_BLOCKING=1`).

---

## Summary of Phase Ordering Implications

Based on the pitfall dependency graph:

1. **Phase 1 (Frontend + Op Signature):** Pits 2, 3, 9, 10 — unblock the ability to BUILD and VERIFY the MLIR op. Must be done first — no other phase can progress without this.

2. **Phase 1 (Hook Integration):** Pit 8 — extend `CudaFuncRequest` and suspended-compiler path for `SharedTensorParameter`. Enables CUDA compilation with shared-memory args.

3. **Phase 2 (Lowering):** Pits 1, 5, 7 — the three address/offset/ordering correctness issues in `ExternCallOpToLLVM`. These are codegen bugs that cause wrong results.

4. **Phase 3 (Verification):** Pits 4, 6, 11 — tested by E2E tests. These are correctness issues that are invisible until runtime and can only be caught by testing.

5. **Phase 4 (Documentation):** Pit 12 — user-facing warning. No code change, just docs.

**Critical dependency:** Phase 1 blocks Phase 2 blocks Phase 3. Phase 1's two sub-parts (op signature + hook integration) can be parallelized. Phase 3 cannot begin until Phase 2 is complete because the lowering must be fixed before tests can pass.

---

## Sources

| File:Line | What it provides | Pitfall |
|-----------|-----------------|---------|
| `ExternCallOpToLLVM.cpp:143-152` | `alloca + store + ptr` (addrspace 0) for every operand | Pit 1 |
| `Utility.h:374-448` | `SharedMemoryObject` with bases (addrspace 3) + offsets | Pit 1, 5 |
| `Utility.cpp:1428-1452` | `getSharedMemoryObjectFromStruct()` unpacks LLVM struct → SharedMemoryObject | Pit 1, 5 |
| `Utility.cpp:1569-1589` | `getSharedMemoryBase()` returns `ptr addrspace(3)` | Pit 1 |
| `TargetInfo.cpp:620` | `getSharedAddressSpace() returns 3` (canonical for NVIDIA) | Pit 1 |
| `clang_compiler.cc:1169-1170` | `cast<RankedTensorType>(operand.getType())` — crashes on MemDescType | Pit 2 |
| `LinearLayoutConversions.cpp:1376-1383` | `toLinearLayout(MemDescType)` uses allocShape | Pit 2, 4 |
| `TritonGPUOps.td:802-808` | `Variadic<TT_Tensor>:$inputs` — rejects MemDescType | Pit 3 |
| `_layouts.py:631-673` | `SharedLinearLayout` Python — offset_bases, block_bases, alignment, shape computation | Pit 4 |
| `Dialect.cpp:2061-2070` | `SharedLinearEncodingAttr::toLinearLayout()` — MLIR offset computation | Pit 4 |
| `gluon_ir.cc:454-466` | `get_shared_linear_layout()` builds `LinearLayout` from offset/block bases | Pit 4 |
| `compiler.py:683-686` | Parse-counter assertion (`parse_count_delta == distinct_cu`) | Pit 8 |
| `clang_compiler.cc:54-57` | `sExternCudaParseCount` static counter + comments | Pit 8 |
| `compiler.py:190-218` | `InferExternCallResult.create_and_suspend()` — suspended compiler creation | Pit 8 |
| `compiler.py:776-806` | Suspended compiler reuse in `_pre_compile_extern_calls()` | Pit 8 |
| `clang_compiler.h:165-169` | `CudaFuncRequest.paramTypes` — variant needs SharedTensorParameter | Pit 8 |
| `_semantic.py:250-318` | `call_extern()` arg validation + arg_params construction | Pit 10 |
| `_core.py:446-475` | `shared_memory_descriptor.slice()` / `.index()` — subview methods | Pit 5 |
| `clang_compiler.cc:1340-1396` | `linkBitcodeToModule` ret-type fix, DISubprogram stripping, alwaysInline | Pit 11 |
| `_core.py:703-714` | `gl.barrier()` CTA-wide sync — user must call manually | Pit 12 |

(End of file — 11 pages)
