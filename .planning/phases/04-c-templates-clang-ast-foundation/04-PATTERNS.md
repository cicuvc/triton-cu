# Phase 04: C++ Templates + Clang AST Foundation - Pattern Map

**Mapped:** 2026-07-12
**Files analyzed:** 5
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `python/test/gluon/tt_plugin.cu` (modified) | device-template | compile-time-constexpr | `TensorLayout<Shape,N_WARPS>::Layout` + `Tensor<T,Shape,Layout>` in same file (lines 30-99) | **exact** — same file, same pattern, shared primitives |
| `python/src/clang_compiler.h` (modified) | header-declaration | request-response AST | `LayoutInfo`/`TensorParameter` (lines 129-141), `TypeBuilder` (lines 224-257), `TypeInspector` (lines 263-280), `FunctionResolver` (lines 286-296) | **exact** — parallel structs mirror existing line-for-line |
| `python/src/clang_compiler.cc` (modified) | implementation | request-response AST | `BuildLayoutFactory`/`BuildBasisGroup`/`BuildLayout`/`BuildTensor`/`DispatchTypeParsing`/`FunctionResolver::LookupFunction` (lines 205-571) | **exact** — parallel implementations mirror existing public methods |
| `python/src/llvm.cc` (modified) | pybind11-binding | request-response | `py::class_<TensorParameter>` (lines 955-1003) | **exact** — `def_property` / `def_readwrite` pattern repeated |
| `python/test/gluon/test_shared_tensor.py` (new) | test | test-verification | `test_extern_call.py` (lines 1-167) minus `@gluon.jit`/GPU | **role-match** — pytest module, different backend (CPU-only, no GPU) |

## Pattern Assignments

### `python/test/gluon/tt_plugin.cu` — SharedLinearLayout + SharedTensor device templates

**Analog:** Existing `IntTuple<N>` / `TensorLayout::BasisGroup` / `Tensor<T,Shape,Layout>` (lines 7-99 same file)

**Shared primitives to reuse (no modification needed):**
- `IntTuple<N>` (lines 15-28) — used as basis row carrier for `OffsetBases`/`BlockBases`
- `Shape<DIMS...>` (lines 7-13) — tensor shape carrier

**BasisGroup::evaluate() pattern to mirror** (lines 45-49):
```cpp
constexpr IntTuple<RANK> evaluate(uint32_t x) const {
    return ([&]<size_t...IDX>(std::index_sequence<IDX...>){ 
        return ((((x >> IDX) & 0x1) ? Dims[IDX] : IntTuple<RANK>{}) + ... + IntTuple<RANK>{}); 
    })(std::make_index_sequence<N_BASES>{});
}
```

**SharedLinearLayout template signature** (D-01, D-02):
```cpp
template<uint32_t RANK>
struct OffsetBases {  // NTTP carrier — each row is IntTuple<RANK>
    static constexpr uint32_t N_BASES = 0;  // filled by class specialization
    // Dims array of IntTuple<RANK>[N_BASES]
};

template<uint32_t RANK>
struct BlockBases {
    static constexpr uint32_t N_BASES = 0;
};

template<typename OffsetBasesT, typename BlockBasesT, uint32_t Alignment>
struct SharedLinearLayout {
    static constexpr uint32_t RANK = OffsetBasesT::RANK;
    static constexpr uint32_t Align = Alignment;

    // OffsetBasesT::Dims[i] = IntTuple<RANK>{...row i...}
    // BlockBasesT::Dims[i] = IntTuple<RANK>{...row i...}

    static constexpr IntTuple<RANK> evaluate(
        const IntTuple<RANK>& logicalIndices,
        const IntTuple<RANK>& blockIndices)  // Phase 4: blockIndices = {}
    {
        // offset = sum over offset bases, block = sum over block bases
        // total = offset + block (element-wise add, like BasisGroup::evaluate but multi-index)
    }
};
```

**SharedTensor template signature** (D-03, D-04):
```cpp
template<typename T, typename TShape, typename TLayout>
struct SharedTensor {
    T data[];  // zero-length array — aliases external shared memory (D-03)

    __device__ T& operator()(/* logical indices ... */) {
        // logicalIndices → layout.evaluate(indices, blockIndices) → offset
        // return data[offset];  // T& (mutable, both read+write) (D-04)
    }
};
```

**Key difference from existing `Tensor` (lines 85-99):**
- `Tensor` stores `T data[TLayout::REG_SIZE]` — value array with owned storage
- `SharedTensor` stores `T data[]` — zero-length base pointer (aliases external shared memory)
- `Tensor` access: `data[i]` directly indexed by register index
- `SharedTensor` access: `operator()(logicalIndices...)` via `layout.evaluate(indices) → offset → data[offset]`

---

### `python/src/clang_compiler.h` — SharedLayoutInfo / SharedTensorParameter + TypeBuilder/TypeInspector extensions

**Analog 1:** `LayoutInfo` struct (lines 129-135) → `SharedLayoutInfo`
```cpp
struct LayoutInfo {              // existing — template for SharedLayoutInfo
  std::vector<uint32_t> LayoutShape;
  std::vector<uint32_t> RegBasis;
  std::vector<uint32_t> LaneBasis;
  std::vector<uint32_t> WarpBasis;
  uint32_t N_WARPS = 0;
};
```

**New struct (SHAST-01):**
```cpp
struct SharedLayoutInfo {
  uint32_t RANK = 0;
  std::vector<uint32_t> OffsetBasis;   // flat: [row0_elem0, row0_elem1, ..., row1_elem0, ...]
  std::vector<uint32_t> BlockBasis;    // flat (may be empty for v1.1)
  uint32_t Alignment = 16;
};
```

**Analog 2:** `TensorParameter` struct (lines 137-141) → `SharedTensorParameter`
```cpp
struct TensorParameter {          // existing — template for SharedTensorParameter
  ScalarType Type;
  std::vector<uint32_t> Shape;
  LayoutInfo Layout;
};
```

**New struct (SHAST-01):**
```cpp
struct SharedTensorParameter {
  ScalarType Type;
  std::vector<uint32_t> Shape;
  SharedLayoutInfo Layout;
};
```

**Analog 3:** `CudaFuncRequest::ParamTypes` variant (line 167) — where `SharedTensorParameter` plugs in:
```cpp
struct CudaFuncRequest {
  std::string Symbol;
  std::vector<std::variant<ScalarType, TensorParameter>> ParamTypes;   // existing
  // becomes: std::vector<std::variant<ScalarType, TensorParameter, SharedTensorParameter>>
  bool UseFastMath = false;
};
```

**Analog 4:** `TypeBuilder` declarations (lines 224-257) → parallel `BuildSharedLinearLayout`/`BuildSharedTensor`:
```cpp
// Existing patterns to mirror:
clang::QualType BuildLayout(const LayoutFactoryContext &LF,
                            clang::TemplateArgument aRegs,
                            clang::TemplateArgument aLanes,
                            clang::TemplateArgument aWarps);
clang::QualType BuildTensor(clang::QualType ElementType,
                            clang::QualType ShapeType,
                            clang::QualType LayoutType,
                            bool instantiate = true);
```

**New declarations (SHAST-02):**
```cpp
// In struct TypeBuilder { ... }:
clang::ClassTemplateDecl *SharedLinearLayoutTemplateType;   // new member
clang::ClassTemplateDecl *SharedTensorTemplateType;         // new member

clang::QualType BuildSharedLinearLayout(const SharedLayoutInfo &info);
clang::QualType BuildSharedTensor(clang::QualType ElementType,
                                  clang::QualType ShapeType,
                                  clang::QualType LayoutType);
```

**Analog 5:** `TypeInspector` declarations (lines 263-280) → parallel `ParseSharedTensorType`:
```cpp
// Existing pattern:
TensorParameter ParseTensorType(clang::ClassTemplateSpecializationDecl *type);
std::variant<std::nullptr_t, TensorParameter, TupleType>
DispatchTypeParsing(clang::QualType type);
```

**New declarations (SHAST-02):**
```cpp
// In struct TypeInspector { ... }:
clang::ClassTemplateDecl *SharedTensorTemplateType;   // new member

SharedTensorParameter ParseSharedTensorType(clang::ClassTemplateSpecializationDecl *type);
// DispatchTypeParsing return type: extend variant OR add parallel branch
// Agent's Discretion: variant<nullptr_t, TensorParameter, SharedTensorParameter, TupleType>
```

**Analog 6:** `FunctionResolver` (lines 286-296) — unchanged interface, but `LookupFunction` must work with `SharedTensor&` param types:
```cpp
// No interface change — LookupFunction already takes ArrayRef<QualType>
clang::FunctionDecl *LookupFunction(const llvm::StringRef &Name,
                                     const llvm::ArrayRef<clang::QualType> &ArgumentTypes);
// SharedTensor& params are just QualType args like any other.
// Sema::DeduceTemplateArguments handles them via standard C++ overload resolution.
```

**Analog 7:** `TensorTypeHelpers` facade (lines 302-309) — gains `SharedTensorTemplateType` pass-through:
```cpp
struct TensorTypeHelpers {
  TypeBuilder Builder;
  TypeInspector Inspector;
  FunctionResolver Resolver;

  TensorTypeHelpers(clang::ASTContext &Ctx, clang::Sema &S);
};
// No structural change needed — Builder/Inspector internals gain shared-tensor awareness
```

---

### `python/src/clang_compiler.cc` — implementations of shared path

**Analog 1:** `TypeBuilder::BuildLayout` (lines 284-304) → `BuildSharedLinearLayout`
```cpp
// Existing pattern:
clang::QualType TypeBuilder::BuildLayout(
    const LayoutFactoryContext &LF, clang::TemplateArgument aRegs,
    clang::TemplateArgument aLanes, clang::TemplateArgument aWarps) {
  auto SL = LayoutFactoryTemplateType->getLocation();
  auto *specArgs = clang::TemplateArgumentList::CreateCopy(
      Ctx, {aRegs, aLanes, aWarps});
  void *ins = nullptr;
  clang::ClassTemplateSpecializationDecl *Spec;
  if (!(Spec = LF.LayoutTmpl->findSpecialization(specArgs->asArray(), ins))) {
    Spec = clang::ClassTemplateSpecializationDecl::Create(
        Ctx, clang::TagTypeKind::Struct, LF.spec, SL, SL,
        LF.LayoutTmpl, specArgs->asArray(), false, nullptr);
    LF.LayoutTmpl->AddSpecialization(Spec, ins);
  }
  return Ctx.getTemplateSpecializationType(
      clang::ElaboratedTypeKeyword::None,
      clang::TemplateName(LF.LayoutTmpl), specArgs->asArray(),
      specArgs->asArray(), Ctx.getCanonicalTagType(Spec));
}
```

**New implementation strategy:** `BuildSharedLinearLayout` is simpler — `SharedLinearLayout<OffsetBases{...}, BlockBases{...}, Align>` is a flat top-level template (D-01). Instead of a LayoutFactory context, build directly:
1. Build `OffsetBases<RANK>` class specialization with `IntTuple<RANK>` array as NTTP value
2. Build `BlockBases<RANK>` class specialization (same pattern)
3. Build `SharedLinearLayout<OffsetBases, BlockBases, Alignment>` specialization
4. Return `QualType` via `getTemplateSpecializationType`

**Analog 2:** `TypeBuilder::BuildTensor` (lines 323-349) → `BuildSharedTensor`
```cpp
// Existing pattern:
clang::QualType TypeBuilder::BuildTensor(clang::QualType ElementType,
                                         clang::QualType ShapeType,
                                         clang::QualType LayoutType,
                                         bool instantiate) {
  auto SL = TensorTemplateType->getLocation();
  auto *args = clang::TemplateArgumentList::CreateCopy(
      Ctx, {mkTypeArg(ElementType), mkTypeArg(ShapeType),
            mkTypeArg(LayoutType)});
  void *ins = nullptr;
  clang::ClassTemplateSpecializationDecl *Spec;
  if (!(Spec = TensorTemplateType->findSpecialization(args->asArray(), ins))) {
    Spec = clang::ClassTemplateSpecializationDecl::Create(
        Ctx, clang::TagTypeKind::Struct, Ctx.getTranslationUnitDecl(),
        SL, SL, TensorTemplateType, args->asArray(), false, nullptr);
    TensorTemplateType->AddSpecialization(Spec, ins);
  }
  if (instantiate && !Spec->hasDefinition())
    SemaRef.InstantiateClassTemplateSpecialization(
        SL, Spec, clang::TSK_ImplicitInstantiation, false, false);
  return Ctx.getTemplateSpecializationType(
      clang::ElaboratedTypeKeyword::Struct,
      clang::TemplateName(TensorTemplateType), args->asArray(),
      args->asArray(), Ctx.getCanonicalTagType(Spec));
}
```

**New implementation:** Identical pattern, but using `SharedTensorTemplateType` and 3 template args:
- Arg 0: scalar `QualType`
- Arg 1: `Shape<DIMS...>` `QualType` (same Shape template — shared with existing `buildShape`)
- Arg 2: `SharedLinearLayout<...>` `QualType`

**Analog 3:** `TypeBuilder::BuildBasisGroup` (lines 239-282) → basis carrier for OffsetBases/BlockBases

The `BuildBasisGroup` builds `BasisGroup<N_BASES>` specializations with `IntTuple<RANK>` as template param objects. This exact pattern is reused for constructing `OffsetBases` and `BlockBases` carrier types, but instead of `BasisGroup`, use a simpler flat carrier:
```cpp
// Build OffsetBases<RANK> with IntTuple<RANK>[N_BASES] array as NTTP
// Build BlockBases<RANK> similarly
```

**Analog 4:** `TypeInspector::DispatchTypeParsing` (lines 453-472) + `ParseTensorType` (lines 428-441) → shared-tensor parse branch
```cpp
// Existing DispatchTypeParsing:
std::variant<std::nullptr_t, TensorParameter, TupleType>
TypeInspector::DispatchTypeParsing(clang::QualType type) {
  if (auto *RecordDecl = type->getAsRecordDecl()) {
    if (auto *ClassSpecDecl =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(RecordDecl)) {
      if (ClassSpecDecl->getSpecializedTemplate() == TensorTemplateType)
        return ParseTensorType(ClassSpecDecl);
      if (ClassSpecDecl->getSpecializedTemplate()->getNameAsString() == "tuple") {
        // ... parse tuple
      }
    }
  }
  return nullptr;
}
```

**New branch:**
```cpp
// Add before the TensorTemplateType check:
if (ClassSpecDecl->getSpecializedTemplate() == SharedTensorTemplateType)
    return ParseSharedTensorType(ClassSpecDecl);
```

**ParseSharedTensorType pattern** (mirrors `ParseTensorType` lines 428-441):
```cpp
SharedTensorParameter TypeInspector::ParseSharedTensorType(
    clang::ClassTemplateSpecializationDecl *type) {
  auto ScalarType = type->getTemplateArgs().get(0).getAsType();        // T
  auto Shape = ParseShapeType(type->getTemplateArgs().get(1).getAsType());  // Shape<...>
  // Arg 2: SharedLinearLayout<OffsetBases, BlockBases, Alignment>
  auto LayoutSpec = dyn_cast<ClassTemplateSpecializationDecl>(
      type->getTemplateArgs().get(2).getAsType()->getAsRecordDecl());
  // Parse offset_bases from NTTP (IntTuple<RANK> arrays)
  // Parse block_bases from NTTP
  // Parse alignment from integral template arg
  SharedTensorParameter tp;
  tp.Type = getScalarTypeFromQualType(Ctx, ScalarType);
  tp.Shape.assign(Shape.begin(), Shape.end());
  tp.Layout.OffsetBasis = /* flat from IntTuple parsing */;
  tp.Layout.BlockBasis = /* flat from IntTuple parsing */;
  tp.Layout.Alignment = /* from template arg */;
  return tp;
}
```

**Analog 5:** `FunctionResolver::LookupFunction` (lines 483-563) — no code change needed.

Works as-is for `SharedTensor&` params. The key insight: `SharedTensor<...>&` is just a `QualType` like any other parameter. `SemaRef.DeduceTemplateArguments` performs standard C++ template argument deduction on whatever types are passed. If a device function is declared as `template<typename T, ...> __device__ void func(SharedTensor<T, Shape<32>, SomeLayout>&)`, the deduction engine resolves `T` from the passed `QualType`.

The only structural requirement: `SharedTensor` must be declared in the CUDA source (tt_plugin.cu) so clang's name lookup finds it.

**Analog 6:** `CUDACompiler::BuildTensor` (lines 705-768) → add `BuildSharedTensor` parallel method (or handle via ParamTypes variant dispatch):
```cpp
// Existing pattern — BuildTensor dispatches a TaskQueue lambda:
clang::QualType CUDACompiler::BuildTensor(const TensorParameter &Param) {
  clang::QualType Result;
  TaskQueue.emplace([&](TensorTypeHelpers &helper, CustomAstConsumer &) {
    auto Shape = helper.Builder.buildShape(Param.Shape);
    // ... PlaceholderLayout fallback, then BuildLayoutFactory → BuildBasisGroup → BuildLayout → BuildTensor
    Result = helper.Builder.BuildTensor(
        getQualTypeFromScalarType(helper.Builder.Ctx, Param.Type),
        Shape.type, Layout);
  });
  InvocationContext->SwitchTo(*CompileExecutionContext);
  return Result;
}
```

**New variant dispatch in `compileBitcode`/`inferReturnTypes`** (lines 950-967):
```cpp
// Existing:
if (auto *tp = std::get_if<TensorParameter>(&req.ParamTypes[J])) {
    argTypes[J] = this->BuildTensor(*tp);
} else if (auto *st = std::get_if<ScalarType>(&req.ParamTypes[J])) {
    // ...
}
// New arm:
if (auto *stp = std::get_if<SharedTensorParameter>(&req.ParamTypes[J])) {
    argTypes[J] = this->BuildSharedTensor(*stp);
}
```

`BuildSharedTensor` parallels `BuildTensor` (lines 705-768) but:
- Calls `helper.Builder.BuildSharedLinearLayout(stp->Layout)` instead of `BuildLayoutFactory` chain
- Calls `helper.Builder.BuildSharedTensor(scalarType, shapeType, layoutType)` instead of `BuildTensor`

---

### `python/src/llvm.cc` — SharedTensorParameter pybind11 binding

**Analog:** `TensorParameter` binding (lines 955-1003)
```cpp
py::class_<TensorParameter>(m, "TensorParameter")
    .def(py::init<>())
    .def_readwrite("type", &TensorParameter::Type)
    .def_readwrite("shape", &TensorParameter::Shape)
    .def_property(
        "layout_shape",
        [](TensorParameter &tp) -> std::vector<uint32_t> & {
          return tp.Layout.LayoutShape;
        },
        [](TensorParameter &tp, std::vector<uint32_t> v) {
          tp.Layout.LayoutShape = std::move(v);
        },
        py::return_value_policy::reference_internal)
    .def_property(
        "reg_basis",
        // ... same pattern ...
    )
    // ... lane_basis, warp_basis, n_warps follow same pattern ...
```

**New binding (place immediately after TensorParameter block, before line 1005):**
```cpp
py::class_<SharedTensorParameter>(m, "SharedTensorParameter")
    .def(py::init<>())
    .def_readwrite("type", &SharedTensorParameter::Type)
    .def_readwrite("shape", &SharedTensorParameter::Shape)
    .def_property(
        "offset_basis",
        [](SharedTensorParameter &stp) -> std::vector<uint32_t> & {
          return stp.Layout.OffsetBasis;
        },
        [](SharedTensorParameter &stp, std::vector<uint32_t> v) {
          stp.Layout.OffsetBasis = std::move(v);
        },
        py::return_value_policy::reference_internal)
    .def_property(
        "block_basis",
        [](SharedTensorParameter &stp) -> std::vector<uint32_t> & {
          return stp.Layout.BlockBasis;
        },
        [](SharedTensorParameter &stp, std::vector<uint32_t> v) {
          stp.Layout.BlockBasis = std::move(v);
        },
        py::return_value_policy::reference_internal)
    .def_property(
        "alignment",
        [](SharedTensorParameter &stp) -> uint32_t & {
          return stp.Layout.Alignment;
        },
        [](SharedTensorParameter &stp, uint32_t v) {
          stp.Layout.Alignment = v;
        },
        py::return_value_policy::reference_internal);
```

**Also add the variant visitor in `compile_cuda_to_module` result unwrapping** (lines 1177-1205). The existing code iterates `r.ReturnTypes` as `TensorParameter`. If `DispatchTypeParsing` can return `SharedTensorParameter`, the result type needs awareness (but for Phase 4, the new CUDA functions return `SharedTensor&` not `SharedTensor` by value — so return type inspection may not need full plumbing yet; see Agent's Discretion scope).

---

### `python/test/gluon/test_shared_tensor.py` — GPU-free pytest harness

**Analog:** `test_extern_call.py` (lines 1-167) minus `@gluon.jit`/GPU

**Pattern to follow for imports and structure:**
```python
import pytest
import triton
from triton._internal_testing import is_cuda  # may not need skipif
```

**Key patterns from test_extern_call.py to mirror:**

1. **Pytest decorator** (line 10):
```python
pytestmark = pytest.mark.skipif(not is_cuda(), reason="CUDA-only test")
# For Phase 4, REMOVE this — tests are GPU-free
```

2. **Test function structure** (lines 73-81):
```python
@pytest.mark.parametrize("BLOCK", [512])
def test_elementwise_add(BLOCK):
    torch.set_default_device('cuda')          # REMOVE for Phase 4
    x = torch.randn((BLOCK,), dtype=torch.float32)
    y = torch.randn((BLOCK,), dtype=torch.float32)
    out = torch.empty_like(x)
    elementwise_add_kernel[(1,)](x, y, out, num_warps=1)  # GPU — NOT Phase 4
    torch.cuda.synchronize()                                # REMOVE
    torch.testing.assert_close(out, x + y)
```

3. **Phase 4 test structure (new, no GPU):**
```python
"""GPU-free unit tests for SharedTensorParameter + clang AST inference.

Verifies (D-08):
  (a) compilation succeeds
  (b) round-tripped SharedTensorParameter matches input
      (scalar type, shape dims, offset/block bases, alignment)
  (c) FunctionResolver resolves without substitution failure/ambiguity
  (d) parity check: evaluate() output matches MLIR LinearLayout
      composition (D-07)
"""

import pytest
from triton._internal_testing import LLVM_CONFIG  # reuse if exists
# Import llvm module for SharedTensorParameter binding


def test_shared_linear_layout_round_trip():
    """SharedLayoutInfo → clang AST → SharedLayoutInfo round-trip."""
    # 1. Construct llvm.SharedTensorParameter in Python
    # 2. Use SuspendedCudaCompiler to parse a synthetic .cu
    # 3. Build SharedTensor via TypeBuilder
    # 4. Parse back via TypeInspector
    # 5. Assert scalar type, shape, offset_basis, block_basis, alignment match
    pass


def test_shared_tensor_function_resolution():
    """FunctionResolver resolves __device__ function with SharedTensor& param."""
    # 1. Synthetic .cu with __device__ func(SharedTensor<float, Shape<32>, SharedLinearLayout<...>>& x)
    # 2. Build arg types → LookupFunction
    # 3. Assert FD != nullptr (no substitution failure)
    pass


def test_swizzle_parity():
    """D-07: evaluate() output matches MLIR LinearLayout({offsetBases, blockBases})."""
    # 1. Define a non-trivial swizzled layout (e.g., 2-rank, 128B swizzle)
    # 2. Compute expected offsets via Python LinearLayout composition
    # 3. Compute C++ evaluate() output (via constexpr or host invocation)
    # 4. Assert bit-identical for all logical index combinations
    pass
```

**Pytest fixture pattern** — from existing test structure, use `conftest.py` or inline:
- No `torch` needed (GPU-free)
- Use `triton` for `llvm` module access if available
- Direct `llvm.SharedTensorParameter()` construction patterns

**Synthetic CUDA source pattern** (for driving clang inference):
```python
SYNTHETIC_CU = """
#include <cstdint>
// ... SharedLinearLayout + SharedTensor definitions (from tt_plugin.cu)

template<typename T, typename TLayout>
__device__ void write_shared(SharedTensor<T, Shape<32>, TLayout>& shm) {
    shm(0) = T{};
}
"""
```

The test constructs a `SuspendedCudaCompiler(source=SYNTHETIC_CU, ...)`, calls `.parse()`, then builds arg types and calls the inference API to verify round-trip correctness — same pattern as `test_gl_call_no_inference_hook_raises` (lines 140-167) but for the clang inference path directly, not via `gl.call`.

---

## Shared Patterns

### Coroutine TaskQueue Pattern (all CUDACompiler methods)

**Source:** `CUDACompiler::BuildTensor` (lines 705-768), `LookupFunction` (lines 780-789), `EvaluateFunctionReturnType` (lines 909-921)

**Pattern:** Every method that needs to run in the clang thread pushes a lambda to `TaskQueue` and then switches context via `InvocationContext->SwitchTo(*CompileExecutionContext)`.

```cpp
clang::QualType CUDACompiler::BuildSharedTensor(const SharedTensorParameter &Param) {
  clang::QualType Result;
  TaskQueue.emplace([&](TensorTypeHelpers &helper, CustomAstConsumer &) {
    // ... build AST types using helper.Builder ...
    Result = helper.Builder.BuildSharedTensor(...);
  });
  InvocationContext->SwitchTo(*CompileExecutionContext);
  return Result;
}
```

**Apply to:** `BuildSharedTensor`, `BuildSharedLinearLayout` — every new CUDACompiler method.

### TemplateSpecializationDecl Create + Specialize + getTemplateSpecializationType Pattern

**Source:** `TypeBuilder::BuildTensor` (lines 323-349), `BuildLayout` (lines 284-304), `BuildInts` (lines 306-321)

**Pattern:** All TypeBuilder methods follow this 5-step pattern:
1. Create `TemplateArgumentList::CreateCopy` with the template args
2. Try `findSpecialization` — if found, reuse
3. If not found, `ClassTemplateSpecializationDecl::Create` + `AddSpecialization`
4. Optionally `SemaRef.InstantiateClassTemplateSpecialization`
5. Return `Ctx.getTemplateSpecializationType`

```cpp
auto *args = clang::TemplateArgumentList::CreateCopy(Ctx, {arg0, arg1, arg2});
void *ins = nullptr;
clang::ClassTemplateSpecializationDecl *Spec;
if (!(Spec = SomeTmpl->findSpecialization(args->asArray(), ins))) {
    Spec = clang::ClassTemplateSpecializationDecl::Create(
        Ctx, TagTypeKind::Struct, parentDecl, SL, SL,
        SomeTmpl, args->asArray(), false, nullptr);
    SomeTmpl->AddSpecialization(Spec, ins);
}
// ... optional InstantiateClassTemplateSpecialization ...
return Ctx.getTemplateSpecializationType(
    ElaboratedTypeKeyword::Struct,
    TemplateName(SomeTmpl), args->asArray(),
    args->asArray(), Ctx.getCanonicalTagType(Spec));
```

**Apply to:** `BuildSharedLinearLayout`, `BuildSharedTensor`.

### NTTP Array Construction Pattern (for OffsetBases/BlockBases)

**Source:** `TypeBuilder::BuildBasisGroup` (lines 239-282)

**Pattern for constructing `IntTuple<RANK>` arrays as template param objects:**
```cpp
llvm::SmallVector<APValue, 4> Elts(N_BASES);
for (unsigned i = 0; i < N_BASES; ++i) {
    Elts[i] = APValue(APValue::UninitStruct(), 0u, 1u);
    Elts[i].getStructField(0) = APValue(APValue::UninitArray(), RANK, RANK);
    for (unsigned r = 0; r < RANK; ++r)
        Elts[i].getStructField(0).getArrayInitializedElt(r) =
            APValue(APSInt(APInt(32, vecs[i * RANK + r], false)));
}
APValue val(APValue::UninitStruct(), 0u, 1u);
val.getStructField(0) = APValue(APValue::UninitArray(), N_BASES, N_BASES);
for (unsigned i = 0; i < N_BASES; ++i)
    val.getStructField(0).getArrayInitializedElt(i) = std::move(Elts[i]);
auto *TPOD = Ctx.getTemplateParamObjectDecl(BGType, val);
return {clang::TemplateArgument(TPOD, Ctx.getCanonicalType(BGType)), S};
```

**Apply to:** OffsetBases/BlockBases NTTP carrier construction in `BuildSharedLinearLayout`.

### DispatchTypeParsing Variant Dispatch Pattern

**Source:** `TypeInspector::DispatchTypeParsing` (lines 453-472)

**Pattern for adding a new type arm:**
```cpp
std::variant<std::nullptr_t, TensorParameter, SharedTensorParameter, TupleType>
TypeInspector::DispatchTypeParsing(clang::QualType type) {
  if (auto *RecordDecl = type->getAsRecordDecl()) {
    if (auto *ClassSpecDecl = dyn_cast<ClassTemplateSpecializationDecl>(RecordDecl)) {
      if (ClassSpecDecl->getSpecializedTemplate() == SharedTensorTemplateType)
        return ParseSharedTensorType(ClassSpecDecl);       // new arm
      if (ClassSpecDecl->getSpecializedTemplate() == TensorTemplateType)
        return ParseTensorType(ClassSpecDecl);             // existing
      if (ClassSpecDecl->getSpecializedTemplate()->getNameAsString() == "tuple") {
        // ...
      }
    }
  }
  return nullptr;
}
```

### Error Handling

**Source:** `CUDACompiler::compileBitcode` (lines 1048-1049)
```cpp
if (!FD)
  return {"", "Function lookup failed: " + req.Symbol, {}};
```

**Pattern:** Return `tuple<string, string, vector<CudaFuncResult>>` where empty second string means no error. All new methods follow this convention.

### Validation (No MLIR in this phase)

Phase 4 validation is test-driven (D-08), not via MLIR assertion. The `SharedTensorParameter` validates at construction time in the test.

---

## No Analog Found

All files have exact or role-match analogs. No files require fallback to RESEARCH.md patterns.

---

## Metadata

**Analog search scope:** `python/test/gluon/`, `python/src/`, `python/triton/experimental/gluon/language/`
**Files scanned:** 12 (tt_plugin.cu, clang_compiler.h, clang_compiler.cc, llvm.cc, test_extern_call.py, cu_compiler_v2.h, cu_compiler_v2.cpp, gluon_ir.cc, _layouts.py, plus 3 additional test files)
**Pattern extraction date:** 2026-07-12
**Canonical refs consumed:** CONTEXT.md §§canonical_refs, AGENTS.md §Extern CUDA C++ Interop, ARCHITECTURE.md §TensorParameter/clang AST Bridge
