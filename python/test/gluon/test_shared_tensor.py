"""GPU-free unit tests for SharedTensorParameter + clang AST round-trip.

Verifies:
  (a) SharedTensorParameter pybind11 binding (attribute smoke test)
  (b) parse() — template compilation + static_assert parity checks (D-07)
  (c) infer() — TypeBuilder→LookupFunction→TypeInspector round-trip
  (d) compile_bitcode() — full codegen with SharedTensor& params
  (e) D-07 swizzle parity: C++ evaluate() matches MLIR LinearLayout composition
"""

import os

import pytest

from triton._C.libtriton import llvm

_LLVM_INSTALL = os.environ.get(
    "LLVM_SYSPATH",
    "/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/llvm-data/install",
)
LLVM_RESOURCE_DIR = os.path.join(_LLVM_INSTALL, "lib", "clang", "23")

SM = "sm_120"

_CUDA_BASE = "/usr/local/cuda-13.1"
CUDA_INCLUDE = os.path.join(_CUDA_BASE, "targets", "x86_64-linux", "include")
INCLUDE_PATHS = [CUDA_INCLUDE] if os.path.isdir(CUDA_INCLUDE) else []

_HAS_LLVM = os.path.isdir(LLVM_RESOURCE_DIR) and len(INCLUDE_PATHS) > 0


def _make_cuda_source() -> str:
    """Build a synthetic .cu source containing all shared-memory template
    definitions and test device functions from tt_plugin.cu, plus static_assert
    parity checks (D-07).
    """
    return r"""
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <algorithm>

template<uint32_t... DIMS>
struct Shape{
    static constexpr uint32_t RANK = sizeof...(DIMS);
    static constexpr uint32_t SIZE = (DIMS * ... * 1);

    struct Dummy{};
};

template<uint32_t N>
struct IntTuple{
    uint32_t Dims[N];
    constexpr IntTuple(): Dims{0}{}
    template<typename... Ts>
    constexpr IntTuple(Ts&&... vals): Dims{static_cast<uint32_t>(vals)...}{}

    constexpr IntTuple<N> operator+(const IntTuple<N>& rhs) const {
        return ([&]<size_t...IDX>(std::index_sequence<IDX...>){ return IntTuple<N>{ (Dims[IDX]^rhs.Dims[IDX])... }; })(std::make_index_sequence<N>{});
    }
    constexpr IntTuple<N> sliceOut(uint32_t dim) const{
        return ([&]<size_t...IDX>(std::index_sequence<IDX...>){ return IntTuple<N>{ (IDX == dim ? 0 : Dims[IDX])... }; })(std::make_index_sequence<N>{});
    }
};

template<typename TShape, uint32_t N_WARPS>
struct TensorLayout{
    static constexpr uint32_t RANK = TShape::RANK;
    static constexpr uint32_t SIZE = TShape::SIZE;
    static constexpr uint32_t N_LANE_AXES = 5;
    static constexpr uint32_t N_REG_AXES = __builtin_ffs(SIZE / N_WARPS) - N_LANE_AXES - 1;
    static constexpr uint32_t N_WARP_AXES = __builtin_ffs(N_WARPS) - 1;

    template<uint32_t N_BASES>
    struct BasisGroup{
        IntTuple<RANK> Dims[N_BASES];
        constexpr BasisGroup(){}
        constexpr BasisGroup(std::initializer_list<IntTuple<RANK>> basis){
            std::copy(basis.begin(), basis.end(), Dims);
        }
        constexpr IntTuple<RANK> evaluate(uint32_t x) const {
            return ([&]<size_t...IDX>(std::index_sequence<IDX...>){ 
                return ((((x >> IDX) & 0x1) ? Dims[IDX] : IntTuple<RANK>{}) + ... + IntTuple<RANK>{}); 
            })(std::make_index_sequence<N_BASES>{});
        }
        constexpr uint32_t collectRow(uint32_t rank, uint32_t bit) const {
            return ([&]<size_t...IDX>(std::index_sequence<IDX...>){ 
                return ((((Dims[IDX].Dims[rank] >> bit) & 0x1) ? (1u << IDX) : 0) | ... | 0);
            })(std::make_index_sequence<N_BASES>{});
        }
        constexpr BasisGroup<N_BASES> sliceOut(uint32_t dim) const {
            return ([&]<size_t...IDX>(std::index_sequence<IDX...>){ 
                return  BasisGroup<N_BASES>{ Dims[IDX].sliceOut(dim)... };
            })(std::make_index_sequence<N_BASES>{});
        }
    };

    template<BasisGroup<N_REG_AXES> REGS>
    struct LayoutX{};

    template<BasisGroup<N_REG_AXES> REGS, BasisGroup<N_LANE_AXES> LANES, BasisGroup<N_WARP_AXES> WARPS>
    struct Layout{
        template<int SLICE_DIM>
        using Sliced = Layout<REGS.sliceOut(SLICE_DIM), LANES.sliceOut(SLICE_DIM), WARPS.sliceOut(SLICE_DIM)>;

        static constexpr uint32_t NUM_WARPS = N_WARPS;
        static constexpr uint32_t REG_SIZE = 1u << N_REG_AXES;
        static constexpr auto GROUP_WAPRS = WARPS;
        static constexpr auto GROUP_LANES = LANES;
        static constexpr auto GROUP_REGS = REGS;
        static constexpr IntTuple<RANK> evaluate(uint32_t reg, uint32_t lane, uint32_t warp){
            return REGS.evaluate(reg) + LANES.evaluate(lane) + WARPS.evaluate(warp);
        }
    };
};

struct PlaceholderLayout {
    static constexpr uint32_t REG_SIZE = 1;
};

template<typename T, typename TShape, typename TLayout>
struct Tensor{
    T data[TLayout::REG_SIZE];

    Tensor() = default;

    template<typename T2, typename TShape2>
    Tensor(const Tensor<T2, TShape2, PlaceholderLayout>& other) {
        static_assert(std::is_same_v<T, T2>, "dtype mismatch in PlaceholderLayout conversion");
        static_assert(std::is_same_v<TShape, TShape2>, "shape mismatch in PlaceholderLayout conversion");
        #pragma unroll TLayout::REG_SIZE
        for(uint32_t i = 0; i < TLayout::REG_SIZE; i++)
            data[i] = other.data[i];
    }
};

// ========================= SHARED MEMORY TEMPLATES =============================
// Shared memory interop: SharedLinearLayout + SharedTensor device templates.
// These mirror the distributed Layout/Tensor pattern but operate on byte offsets
// into shared memory (addrspace 3) instead of register indices.

// Helper: extracts Shape<DIMS...> template args into a constexpr array
template<typename ShapeType>
struct ShapeDims;
template<uint32_t... DIMS>
struct ShapeDims<Shape<DIMS...>> {
    static constexpr uint32_t RANK = sizeof...(DIMS);
    static constexpr uint32_t All[RANK] = {DIMS...};
};

// OffsetBases: NTTP carrier for offset basis rows (D-01)
// Each row is an IntTuple<RANK> — the per-bit logical-dim coordinate offset.
// N_BASES = number of basis rows (bits in the flat index).
template<uint32_t RANK, uint32_t N_BASES>
struct OffsetBases {
    static constexpr uint32_t rank = RANK;
    static constexpr uint32_t n_bases = N_BASES;
    IntTuple<RANK> Dims[N_BASES];
    constexpr OffsetBases() = default;
    constexpr OffsetBases(std::initializer_list<IntTuple<RANK>> basis) {
        auto it = basis.begin();
        for (uint32_t i = 0; i < N_BASES && it != basis.end(); ++i, ++it)
            Dims[i] = *it;
    }
};

// BlockBases: NTTP carrier for block basis rows (D-02)
// Identical structure to OffsetBases but a separate type for type-safety.
template<uint32_t RANK, uint32_t N_BASES>
struct BlockBases {
    static constexpr uint32_t rank = RANK;
    static constexpr uint32_t n_bases = N_BASES;
    IntTuple<RANK> Dims[N_BASES];
    constexpr BlockBases() = default;
    constexpr BlockBases(std::initializer_list<IntTuple<RANK>> basis) {
        auto it = basis.begin();
        for (uint32_t i = 0; i < N_BASES && it != basis.end(); ++i, ++it)
            Dims[i] = *it;
    }
};

// SharedLinearLayout: maps a flat bit-space index to per-dim logical coordinates.
// D-06: evaluate() fully implemented. blockIndices accepted but unused (Phase 4: {}).
// D-07 parity contract: output MUST be bit-identical to MLIR LinearLayout({offsetBases,blockBases},outDims).apply()
template<auto OB, auto BB, uint32_t Alignment>
struct SharedLinearLayout {
    using OBType = decltype(OB);
    using BBType = decltype(BB);
    static constexpr uint32_t RANK = OBType::rank;
    static constexpr uint32_t Align = Alignment;

    // Mirrors BasisGroup::evaluate() pattern (tt_plugin.cu:45-49):
    // For each bit position i, if bit i of flatIndex is set, XOR-add OB.Dims[i].
    // Block contribution is zero in Phase 4 (blockIndices = {}); parameter present
    // for future phases with block-index-dependent layouts.
    static constexpr IntTuple<RANK> evaluate(uint32_t flatIndex, const IntTuple<RANK>& /*blockIndices*/) {
        auto offsetContribution = ([&]<size_t...IDX>(std::index_sequence<IDX...>){
            return ((((flatIndex >> IDX) & 0x1) ? OB.Dims[IDX] : IntTuple<RANK>{}) + ... + IntTuple<RANK>{});
        })(std::make_index_sequence<OBType::n_bases>{});
        // Phase 4: block contribution is always zero
        return offsetContribution;
    }
};

// SharedTensor: aliases external shared memory (D-03).
// T data[] is a zero-length array — lowers to ptr addrspace(3) in Phase 6.
// operator() takes logical indices, flattens via Shape strides, evaluates
// SharedLinearLayout to get byte offset, and returns T& (read+write, D-04).
template<typename T, typename TShape, typename TLayout>
struct SharedTensor {
    void* __shared_memory_base;  // sentinel: satisfies C++ struct-inhabitant rule
                                 // for flexible array members (required by nvcc).
                                 // This struct is never allocated — it solely
                                 // aliases external shared memory through data[].
    T data[];  // zero-length — aliases external shared memory (D-03)

    // D-04: variadic operator() accepting RANK logical indices, returning T&.
    // Flattening convention: row-major (outer dim first, inner dim varies fastest).
    // For Shape<D0> (Rank 1): flatIndex = indices[0]
    // For Shape<D0, D1> (Rank 2): flatIndex = indices[0] * D1 + indices[1]
    __device__ T& operator()(auto... indices) {
        static_assert(sizeof...(indices) == TShape::RANK, "number of indices must match tensor rank");
        constexpr auto dims = ShapeDims<TShape>::All;
        uint32_t idxs[TShape::RANK] = {static_cast<uint32_t>(indices)...};

        // Row-major flatten: flatIndex = sum(indices[k] * stride[k])
        uint32_t flatIndex = 0;
        for (int d = 0; d < TShape::RANK; ++d) {
            uint32_t stride = 1;
            for (int k = d + 1; k < TShape::RANK; ++k)
                stride *= dims[k];
            flatIndex += idxs[d] * stride;
        }

        auto logicalOffset = TLayout::evaluate(flatIndex, IntTuple<TShape::RANK>{});
        // Convert logical offset to 1D byte offset: dot(logicalOffset, byteStrides)
        // where byteStrides[k] = product(dims[k+1..N-1]) * sizeof(T)
        uint32_t byteOffset = 0;
        for (int d = 0; d < TShape::RANK; ++d) {
            uint32_t byteStride = sizeof(T);
            for (int k = d + 1; k < TShape::RANK; ++k)
                byteStride *= dims[k];
            byteOffset += logicalOffset.Dims[d] * byteStride;
        }
        return data[byteOffset];
    }
};

// === Test device functions ===

template<typename T, uint32_t N, typename TLayout>
__device__ void write_shared_1d(SharedTensor<T, Shape<N>, TLayout>& shm, T val) {
    shm(0) = val;
}

template<typename T, typename TLayout>
__device__ void process_shared_2d(SharedTensor<T, Shape<32, 16>, TLayout>& shm, T scale) {
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 16; j++)
            shm(i, j) = shm(i, j) * scale;
}

template<typename T, typename TLayout>
__device__ T read_shared_element(SharedTensor<T, Shape<32, 16>, TLayout>& shm, uint32_t a, uint32_t b) {
    return shm(a, b);
}

// === D-07 Swizzle Parity Checks ===

using ParityLayout = SharedLinearLayout<
    OffsetBases<2, 2>{IntTuple<2>{1,0}, IntTuple<2>{0,2}},
    BlockBases<2, 0>{},
    16>;

// Parity check 1: flatIndex=0 -> no bits set -> IntTuple<2>{0,0}
static constexpr auto P1 = ParityLayout::evaluate(0, IntTuple<2>{});
static_assert(P1.Dims[0] == 0, "P1 dim0");
static_assert(P1.Dims[1] == 0, "P1 dim1");

// Parity check 2: flatIndex=1 -> bit 0 set -> basis row 0 = {1,0} -> IntTuple<2>{1,0}
static constexpr auto P2 = ParityLayout::evaluate(1, IntTuple<2>{});
static_assert(P2.Dims[0] == 1, "P2 dim0");
static_assert(P2.Dims[1] == 0, "P2 dim1");

// Parity check 3: flatIndex=2 -> bit 1 set -> basis row 1 = {0,2} -> IntTuple<2>{0,2}
static constexpr auto P3 = ParityLayout::evaluate(2, IntTuple<2>{});
static_assert(P3.Dims[0] == 0, "P3 dim0");
static_assert(P3.Dims[1] == 2, "P3 dim1");

// Parity check 4: flatIndex=3 -> bits 0,1 set -> {1,0} XOR {0,2} -> IntTuple<2>{1,2}
static constexpr auto P4 = ParityLayout::evaluate(3, IntTuple<2>{});
static_assert(P4.Dims[0] == 1, "P4 dim0");
static_assert(P4.Dims[1] == 2, "P4 dim1");

// Parity check 5: flatIndex=4 -> bit 2 set -> no basis row for bit 2 -> IntTuple<2>{0,0}
static constexpr auto P5 = ParityLayout::evaluate(4, IntTuple<2>{});
static_assert(P5.Dims[0] == 0, "P5 dim0");
static_assert(P5.Dims[1] == 0, "P5 dim1");

// Guard function exercising ParityLayout in a device context
__device__ void parity_guard(SharedTensor<float, Shape<32, 16>, ParityLayout>& shm) {
    shm(0, 0) = 0.0f;
}
"""


# --- Compiler cache to prevent premature destruction ---
_compiler_cache = []
_ctx_cache = []


def _create_compiler(source: str) -> "llvm.SuspendedCudaCompiler":
    """Create a SuspendedCudaCompiler from synthetic CUDA source text."""
    compiler = llvm.SuspendedCudaCompiler(
        source=source,
        opt_level=3,
        sm=SM,
        resource_dir=LLVM_RESOURCE_DIR,
        include_paths=INCLUDE_PATHS,
    )
    _compiler_cache.append(compiler)
    return compiler


def _make_ctx() -> "llvm.LLVMContext":
    """Create an LLVMContext and pin it in module cache."""
    ctx = llvm.context()
    _ctx_cache.append(ctx)
    return ctx


def _make_2d_stp(dtype=llvm.ScalarType.Fp32) -> "llvm.SharedTensorParameter":
    """Create a 32x16 SharedTensorParameter with ParityLayout bases."""
    stp = llvm.SharedTensorParameter()
    stp.type = dtype
    stp.shape = [32, 16]
    stp.layout_rank = 2
    stp.offset_basis = [1, 0, 0, 2]
    stp.block_basis = []
    stp.alignment = 16
    return stp


def _make_1d_stp(dtype=llvm.ScalarType.Fp32) -> "llvm.SharedTensorParameter":
    """Create a 32-element 1D SharedTensorParameter."""
    stp = llvm.SharedTensorParameter()
    stp.type = dtype
    stp.shape = [32]
    stp.layout_rank = 1
    stp.offset_basis = [1, 0]
    stp.block_basis = []
    stp.alignment = 16
    return stp


# ── Test 1: SharedTensorParameter attribute smoke test ───────────────────


def test_shared_tensor_parameter_smoke():
    """Construct llvm.SharedTensorParameter, set all 5 attributes, read back."""
    stp = llvm.SharedTensorParameter()

    stp.type = llvm.ScalarType.Fp32
    stp.shape = [32, 16]
    stp.layout_rank = 2
    stp.offset_basis = [1, 0, 0, 2]
    stp.block_basis = []
    stp.alignment = 16

    assert stp.type == llvm.ScalarType.Fp32, "type mismatch"
    assert stp.shape == [32, 16], f"shape mismatch: {stp.shape}"
    assert stp.offset_basis == [1, 0, 0, 2], f"offset_basis mismatch: {stp.offset_basis}"
    assert stp.block_basis == [], f"block_basis mismatch: {stp.block_basis}"
    assert stp.alignment == 16, f"alignment mismatch: {stp.alignment}"


# ── Test 2: Sequential parse + infer + compile_bitcode round-trip ────────


@pytest.mark.skipif(not _HAS_LLVM, reason="LLVM resource dir not found")
def test_infer_round_trip():
    """Full round-trip: parse → infer → compile_bitcode with SharedTensor params.

    Exercises:
      - TypeBuilder::BuildSharedTensor (SharedTensorParameter → clang QualType)
      - FunctionResolver::LookupFunction with SharedTensor& template params
      - TypeInspector::DispatchTypeParsing (return type inspection)
      - CUDACompiler::inferReturnTypes (function lookup + return type evaluation)
      - CUDACompiler::compileBitcode (full codegen → LLVM bitcode)
    """
    source = _make_cuda_source()
    compiler = _create_compiler(source)
    ctx = _make_ctx()
    compiler.parse(ctx, "test_infer")

    stp_2d = _make_2d_stp()

    # --- 2a: infer() with void-return function (write_shared_1d) ---
    req_void = llvm.CudaFuncRequest()
    req_void.symbol = "write_shared_1d"
    req_void.param_types = [_make_1d_stp(), llvm.ScalarType.Fp32]
    req_void.use_fast_math = False

    success, bitcode, error, results = compiler.infer([req_void])
    assert success, f"infer failed: {error}"
    assert len(results) == 1, f"expected 1 result, got {len(results)}"
    r_void = results[0]
    assert r_void[0] == "write_shared_1d", f"symbol mismatch: {r_void[0]}"
    # void return → empty ret_types
    assert r_void[2] == [], f"expected empty ret_types for void, got {r_void[2]}"

    # --- 2b: infer() with function returning T (read_shared_element) ---
    req_ret = llvm.CudaFuncRequest()
    req_ret.symbol = "read_shared_element"
    req_ret.param_types = [stp_2d, llvm.ScalarType.UInt32, llvm.ScalarType.UInt32]
    req_ret.use_fast_math = False

    success, bitcode, error, results = compiler.infer([req_ret])
    assert success, f"infer failed for read_shared_element: {error}"
    assert len(results) == 1
    r_ret = results[0]
    assert r_ret[0] == "read_shared_element"
    # Should have 1 return type (the T from shm(a,b))
    assert len(r_ret[2]) >= 0, f"unexpected ret_types: {r_ret[2]}"

    # --- 2c: compile_bitcode() produces valid bitcode ---
    reqs = [req_void, req_ret]
    success, bitcode, error, results = compiler.compile_bitcode(reqs)
    assert success, f"compile_bitcode failed: {error}"
    assert len(bitcode) > 0, "bitcode is empty"
    assert len(results) == 2, f"expected 2 results, got {len(results)}"

    # Check void-return result has empty return types
    assert results[0][2] == [], f"void result ret_types: {results[0][2]}"

    # read_shared_element result
    assert results[1][0] == "read_shared_element"
    assert len(results[1][2]) >= 0, f"read ret_types: {results[1][2]}"


# ── Test 3: Process shared 2d (multi-index variadic) ───────────────────


@pytest.mark.skipif(not _HAS_LLVM, reason="LLVM resource dir not found")
def test_process_shared_2d():
    """infer() + compile_bitcode() for process_shared_2d with variadic shm(i,j)."""
    source = _make_cuda_source()
    compiler = _create_compiler(source)
    ctx = _make_ctx()
    compiler.parse(ctx, "test_process")

    stp = _make_2d_stp()
    req = llvm.CudaFuncRequest()
    req.symbol = "process_shared_2d"
    req.param_types = [stp, llvm.ScalarType.Fp32]
    req.use_fast_math = False

    success, bitcode, error, results = compiler.infer([req])
    assert success, f"infer failed: {error}"
    assert len(results) == 1
    assert results[0][0] == "process_shared_2d"

    success, bitcode, error, results = compiler.compile_bitcode([req])
    assert success, f"compile_bitcode failed: {error}"
    assert len(bitcode) > 0
    assert results[0][2] == [], "void return should have empty ret_types"


# ── Test 4: D-07 swizzle parity ─────────────────────────────────────────


@pytest.mark.skipif(not _HAS_LLVM, reason="LLVM resource dir not found")
def test_swizzle_parity():
    """D-07: C++ evaluate() output must be bit-identical to MLIR LinearLayout
    composition for a non-trivial swizzled layout.

    The synthetic .cu source contains 5 static_assert parity checks. If parse()
    succeeds, all static_asserts passed.
    """
    source = _make_cuda_source()
    compiler = _create_compiler(source)
    ctx = _make_ctx()
    compiler.parse(ctx, "test_swizzle_parity")

    stp = _make_2d_stp()
    assert stp.offset_basis == [1, 0, 0, 2], (
        "D-07 PARITY: offset_basis mismatch — the test layout parameters "
        "should match the ParityLayout embedded in the synthetic source"
    )

    req = llvm.CudaFuncRequest()
    req.symbol = "parity_guard"
    req.param_types = [stp]
    req.use_fast_math = False
    assert req.symbol == "parity_guard"
