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

template<typename T, typename TShape, typename TLayout>
struct Tensor{
    T data[TLayout::REG_SIZE];
};


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


using LA = TensorLayout<Shape<512>, 1>::Layout<{{1},{2},{4},{8}}, {{16},{32},{64},{128},{256}}, {}>;

using TS = TensorLayout<Shape<512>, 1>;

template<TS::BasisGroup<2> X>struct BB{};
void fxaa(BB<{{1},{2}}>){}

template
__device__ Tensor<float, Shape<512>, LA> elementwise_add(const Tensor<float, Shape<512>, LA>& lhs, const Tensor<float, Shape<512>, LA>& rhs);