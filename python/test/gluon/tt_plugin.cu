#include <cstdint>
#include <initializer_list>
#include <utility>
#include <algorithm>
#include <tuple>

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

// Test device functions (D-05 exercise) — consumed by Plan 04-03 pytest harness.

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

template<typename T, uint32_t N, typename SharedTLayout, typename TLayout>
__device__ void shared_accumulate(
    SharedTensor<T, Shape<N>, SharedTLayout>& shm,
    const Tensor<T, Shape<N>, TLayout>& val)
{
    #pragma unroll TLayout::REG_SIZE
    for (uint32_t i = 0; i < TLayout::REG_SIZE; i++)
        shm(i) += val.data[i];
}

template<typename T, typename TLayout>
__device__ void write_swizzled_2d(SharedTensor<T, Shape<32, 16>, TLayout>& shm) {
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 16; j++)
            shm(i, j) = static_cast<T>(i * 16 + j);
}

// ========================= END OF DEFINITIONS =============================

template<typename T, uint32_t TILE_WIDTH, typename TLayout>
__device__ Tensor<T, Shape<TILE_WIDTH>, TLayout> elementwise_add(const Tensor<T, Shape<TILE_WIDTH>, TLayout>& lhs, const Tensor<T, Shape<TILE_WIDTH>, TLayout>& rhs){
    Tensor<T, Shape<TILE_WIDTH>, TLayout> result;
    #pragma unroll TLayout::REG_SIZE
    for(int i = 0; i < TLayout::REG_SIZE; i++) result.data[i] = lhs.data[i] + rhs.data[i];
    return result;
}


constexpr uint32_t lowbit(uint32_t x){ return x & (-x); }

template<typename T, uint32_t TILE_WIDTH, typename TLayout>
__device__ Tensor<T, Shape<TILE_WIDTH>, TLayout> intra_warp_add_sibling(const Tensor<T, Shape<TILE_WIDTH>, TLayout>& input){
    // example: input = [a0, b0, a1, b1, ...]
    // result = [a0 + b0, a0 + b0, a1 + b1, a1 + b1, ...]
    
    Tensor<T, Shape<TILE_WIDTH>, TLayout> result = input;

    static_assert(!TLayout::GROUP_WAPRS.collectRow(0, 0), "Inter-warp case not supported");
    constexpr uint32_t reg_mask = lowbit(TLayout::GROUP_REGS.collectRow(0, 0));
    constexpr uint32_t thread_mask = lowbit(TLayout::GROUP_LANES.collectRow(0, 0));
    #pragma unroll TLayout::REG_SIZE
    for(uint32_t i = 0; i < TLayout::REG_SIZE; i++) {
        T remote_val = input.data[i]; //__shfl_xor_sync(~0x0, input.data[i], thread_mask);
        result.data[i ^ reg_mask] += remote_val;
    }
    return result;
}

template<typename T, uint32_t TILE_ROWS, uint32_t TILE_COLS, typename TMatLayout>
__device__ Tensor<T, Shape<TILE_ROWS, TILE_COLS>, TMatLayout> add_bias(const Tensor<T, Shape<TILE_ROWS, TILE_COLS>, TMatLayout>& mat, const Tensor<T, Shape<1, TILE_COLS>, typename TMatLayout::template Sliced<0>>& bias){
    return mat; // not implemented yet
}

using TArg = typename TensorLayout<Shape<32, 32>, 1>::Layout<{{0,1},{0,2},{0,4},{0,8},{0,16}},{{1,0},{2,0},{4,0},{8,0},{16,0}},{}>;
using TRes = typename TensorLayout<Shape<32>, 1>::Layout<{},{{1},{2},{4},{8},{16}},{}>;


template<typename T>
__device__ Tensor<T, Shape<32>, TRes> reduce(const Tensor<T, Shape<32, 32>, TArg>& Vals){
    Tensor<T, Shape<32>, TRes>  Result;
    Result.data[0] = T{};
    #pragma unroll
    for(int i = 0; i < 32; i++){
        Result.data[0] += Vals.data[i];
    }
    return Result;
}

// f16 -> f32 reduction: same layout structure as reduce, different element types
template<typename T>
__device__ Tensor<float, Shape<32>, TRes> reduce_f16(const Tensor<T, Shape<32, 32>, TArg>& Vals){
    Tensor<float, Shape<32>, TRes>  Result;
    Result.data[0] = float{};
    #pragma unroll
    for(int i = 0; i < 32; i++){
        Result.data[0] += Vals.data[i];
    }
    return Result;
}

template<typename T, uint32_t TILE_WIDTH, typename TLayout>
__device__ std::tuple<Tensor<T, Shape<TILE_WIDTH>, TLayout>,
                       Tensor<T, Shape<TILE_WIDTH>, TLayout>>
split_add(const Tensor<T, Shape<TILE_WIDTH>, TLayout>& a,
          const Tensor<T, Shape<TILE_WIDTH>, TLayout>& b) {
    Tensor<T, Shape<TILE_WIDTH>, TLayout> sum, diff;
    #pragma unroll TLayout::REG_SIZE
    for(int i = 0; i < TLayout::REG_SIZE; i++) {
        sum.data[i] = a.data[i] + b.data[i];
        diff.data[i] = a.data[i] - b.data[i];
    }
    return {sum, diff};
}

template<uint32_t N>
struct Ints{};

template<uint32_t N, typename... Ts>
__device__ auto get_tuple_elem(const Ints<N>&, const std::tuple<Ts...>& V){
    return std::get<N>(V);
}