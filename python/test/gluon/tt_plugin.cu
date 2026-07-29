#include <cstdint>
#include <bit>
#include <cassert>
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
    __forceinline__ constexpr IntTuple(): Dims{0}{}
    template<typename... Ts>
    __forceinline__ constexpr IntTuple(Ts&&... vals): Dims{static_cast<uint32_t>(vals)...}{}

    constexpr IntTuple<N> operator+(const IntTuple<N>& rhs) const {
        return ([&]<size_t...IDX>(std::index_sequence<IDX...>){ return IntTuple<N>{ (Dims[IDX]^rhs.Dims[IDX])... }; })(std::make_index_sequence<N>{});
    }
    constexpr IntTuple<N> sliceOut(uint32_t dim) const{
        return ([&]<size_t...IDX>(std::index_sequence<IDX...>){ return IntTuple<N>{ (IDX == dim ? 0 : Dims[IDX])... }; })(std::make_index_sequence<N>{});
    }
};

// Helper: extracts Shape<DIMS...> template args into a constexpr array
template<typename ShapeType>
struct ShapeDims;
template<uint32_t... DIMS>
struct ShapeDims<Shape<DIMS...>> {
    static constexpr uint32_t RANK = sizeof...(DIMS);
    static constexpr uint32_t All[RANK] = {DIMS...};
};

// BasisGroup: NTTP carrier for one basis group (register / lane / warp) of a
// distributed linear layout. A linear layout is a pure linear map — it is
// fully described by its basis rows and does NOT own a shape; the shape
// belongs to the Tensor. All structural constants (REG_SIZE, NUM_WARPS, ...)
// derive from the basis row counts, never from any shape.
template<uint32_t RANK, uint32_t N_BASES>
struct BasisGroup{
    static constexpr uint32_t rank = RANK;
    static constexpr uint32_t n_bases = N_BASES;
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
    constexpr BasisGroup<RANK, N_BASES> sliceOut(uint32_t dim) const {
        return ([&]<size_t...IDX>(std::index_sequence<IDX...>){
            return  BasisGroup<RANK, N_BASES>{ Dims[IDX].sliceOut(dim)... };
        })(std::make_index_sequence<N_BASES>{});
    }
};

// Bounds check for one basis group against a shape's dims.
// Exactness: Triton tensor dims are powers of two, so the XOR-span of
// in-bounds rows stays in-bounds (XOR closure) — per-row checking suffices.
template<uint32_t RANK, uint32_t N_BASES>
constexpr bool groupInBounds(const BasisGroup<RANK, N_BASES>& g, const uint32_t (&dims)[RANK]) {
    for (uint32_t i = 0; i < N_BASES; i++)
        for (uint32_t r = 0; r < RANK; r++)
            if (g.Dims[i].Dims[r] >= dims[r]) return false;
    return true;
}

// TensorLayout: namespace shell for the distributed Layout type.
// Layout<REGS, LANES, WARPS> is a pure linear map from (reg, lane, warp)
// index bits to logical tensor coordinates — no shape, no N_WARPS template
// parameters (NUM_WARPS derives from the warp basis row count).
struct TensorLayout{
    template<auto REGS, auto LANES, auto WARPS>
    struct Layout{
        using RegGroup = decltype(REGS);
        using LaneGroup = decltype(LANES);
        using WarpGroup = decltype(WARPS);
        static_assert(RegGroup::rank == LaneGroup::rank && LaneGroup::rank == WarpGroup::rank,
                      "reg/lane/warp basis groups must have the same rank");

        template<int SLICE_DIM>
        using Sliced = Layout<REGS.sliceOut(SLICE_DIM), LANES.sliceOut(SLICE_DIM), WARPS.sliceOut(SLICE_DIM)>;

        static constexpr uint32_t RANK = RegGroup::rank;
        static constexpr uint32_t N_REG_AXES = RegGroup::n_bases;
        static constexpr uint32_t N_LANE_AXES = LaneGroup::n_bases;
        static constexpr uint32_t N_WARP_AXES = WarpGroup::n_bases;
        static constexpr uint32_t NUM_WARPS = 1u << N_WARP_AXES;
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

// Hardware named barrier handle, passed from Gluon via gl.call as
// `const NamedBarrier&`. The barrier id and arrival count are compile-time
// constants assigned by the Triton named barrier allocation pass and
// materialized into this struct by the extern_call lowering.
//
// NOTE: device functions must NOT call __syncthreads() directly — use
// bar.sync()/bar.arrive() on the barrier the caller handed you, so the
// synchronization scope (which threads participate) stays caller-controlled.
struct NamedBarrier {
    int id;
    int count;

    // Arrive and wait: PTX `bar.sync id, count`.
    __device__ __forceinline__ void sync() const {
        asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count) : "memory");
    }

    // Arrive without waiting: PTX `bar.arrive id, count`. For
    // producer/consumer patterns; consumers wait via sync().
    __device__ __forceinline__ void arrive() const {
        asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(count) : "memory");
    }
};

// Detects concrete TensorLayout::Layout types (vs. PlaceholderLayout probes,
// which carry no basis groups and skip consistency checks).
template<typename L>
concept ConcreteLayout = requires {
    L::RANK;
    L::GROUP_REGS;
    L::GROUP_LANES;
    L::GROUP_WAPRS;
};

template<typename T, typename TShape, typename TLayout>
struct Tensor{
    // Consistency checks between the tensor's shape and the layout's bases.
    // Skipped for PlaceholderLayout probes via if constexpr.
    static constexpr bool LAYOUT_OK = []{
        if constexpr (ConcreteLayout<TLayout>) {
            return TShape::RANK == TLayout::RANK &&
                   groupInBounds(TLayout::GROUP_REGS, ShapeDims<TShape>::All) &&
                   groupInBounds(TLayout::GROUP_LANES, ShapeDims<TShape>::All) &&
                   groupInBounds(TLayout::GROUP_WAPRS, ShapeDims<TShape>::All);
        } else {
            return true;
        }
    }();
    static_assert(LAYOUT_OK,
                  "layout basis rows must have the tensor shape's rank and stay within shape dims");

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

template<typename T>
struct shared_accessor{
    uint32_t __shared_memory_base;

    __forceinline__ __device__ T load() const {
        if constexpr(sizeof(T) == 1){
            uint16_t data;
            asm volatile("ld.shared.b8 %0, [%1];\n":"=h"(data):"r"(__shared_memory_base):"memory");
            uint8_t raw = static_cast<uint8_t>(data);
            return std::bit_cast<T>(raw);
        } else if constexpr(sizeof(T) == 2){
            uint16_t data;
            asm volatile("ld.shared.b16 %0, [%1];\n":"=h"(data):"r"(__shared_memory_base):"memory");
            return std::bit_cast<T>(data);
        } else if constexpr(sizeof(T) == 4){
            uint32_t data;
            asm volatile("ld.shared.b32 %0, [%1];\n":"=r"(data):"r"(__shared_memory_base):"memory");
            return std::bit_cast<T>(data);
        } else if constexpr(sizeof(T) == 8){
            uint64_t data;
            asm volatile("ld.shared.b64 %0, [%1];\n":"=l"(data):"r"(__shared_memory_base):"memory");
            return std::bit_cast<T>(data);
        } else {
            static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                          "shared_accessor: element width must be 1, 2, 4 or 8 bytes");
        }
    }

    __forceinline__ __device__ void store(const T& data) {
        if constexpr(sizeof(T) == 1){
            uint16_t data_u16 = static_cast<uint8_t>(std::bit_cast<uint8_t>(data));
            asm volatile("st.shared.b8 [%1], %0;\n"::"h"(data_u16),"r"(__shared_memory_base):"memory");
        } else if constexpr(sizeof(T) == 2){
            uint16_t data_u16 = std::bit_cast<uint16_t>(data);
            asm volatile("st.shared.b16 [%1], %0;\n"::"h"(data_u16),"r"(__shared_memory_base):"memory");
        } else if constexpr(sizeof(T) == 4){
            uint32_t data_u32 = std::bit_cast<uint32_t>(data);
            asm volatile("st.shared.b32 [%1], %0;\n"::"r"(data_u32),"r"(__shared_memory_base):"memory");
        } else if constexpr(sizeof(T) == 8){
            uint64_t data_u64 = std::bit_cast<uint64_t>(data);
            asm volatile("st.shared.b64 [%1], %0;\n"::"l"(data_u64),"r"(__shared_memory_base):"memory");
        }
    }

    __device__ __forceinline__ operator T() const { return load(); }
    __device__ __forceinline__ T operator =(const T& value) { store(value); return value; }
};

// Vectorized shared-memory primitives: N_ELEMS*sizeof(T) ∈ {4, 8, 16} bytes
// mapped to b32 / v2.b32 / v4.b32 accesses. The caller guarantees natural
// alignment (checked via __trap) and element contiguity (debug assert).
template<typename T, uint32_t N_ELEMS>
__forceinline__ __device__ void shared_load_vector(uint32_t byteAddr, T (&out)[N_ELEMS]) {
    constexpr uint32_t N_BYTES = N_ELEMS * sizeof(T);
    static_assert(N_BYTES == 4 || N_BYTES == 8 || N_BYTES == 16,
                  "shared_load_vector: width must be 4, 8 or 16 bytes");
    constexpr uint32_t N_U32 = N_BYTES / 4;
    uint32_t words[N_U32];
    if constexpr (N_U32 == 1) {
        asm volatile("ld.shared.b32 %0, [%1];\n":"=r"(words[0]):"r"(byteAddr):"memory");
    } else if constexpr (N_U32 == 2) {
        asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n":"=r"(words[0]),"=r"(words[1]):"r"(byteAddr):"memory");
    } else {
        asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n":"=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3]):"r"(byteAddr):"memory");
    }
    __builtin_memcpy(out, words, N_BYTES);
}

template<typename T, uint32_t N_ELEMS>
__forceinline__ __device__ void shared_store_vector(uint32_t byteAddr, const T (&in)[N_ELEMS]) {
    constexpr uint32_t N_BYTES = N_ELEMS * sizeof(T);
    static_assert(N_BYTES == 4 || N_BYTES == 8 || N_BYTES == 16,
                  "shared_store_vector: width must be 4, 8 or 16 bytes");
    constexpr uint32_t N_U32 = N_BYTES / 4;
    uint32_t words[N_U32];
    __builtin_memcpy(words, in, N_BYTES);
    if constexpr (N_U32 == 1) {
        asm volatile("st.shared.b32 [%1], %0;\n"::"r"(words[0]),"r"(byteAddr):"memory");
    } else if constexpr (N_U32 == 2) {
        asm volatile("st.shared.v2.b32 [%2], {%0, %1};\n"::"r"(words[0]),"r"(words[1]),"r"(byteAddr):"memory");
    } else {
        asm volatile("st.shared.v4.b32 [%4], {%0, %1, %2, %3};\n"::"r"(words[0]),"r"(words[1]),"r"(words[2]),"r"(words[3]),"r"(byteAddr):"memory");
    }
}

// cp.async primitives (global → shared, 4/8/16 bytes per thread).
__forceinline__ __device__ void cp_async_ca(uint32_t dstShared, const void* srcGlobal, uint32_t nBytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"::"r"(dstShared),"l"(srcGlobal),"r"(nBytes):"memory");
}
__forceinline__ __device__ void cp_async_cg_16(uint32_t dstShared, const void* srcGlobal) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(dstShared),"l"(srcGlobal):"memory");
}
__forceinline__ __device__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n":::"memory");
}
template<uint32_t N>
__forceinline__ __device__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory");
}

// SharedTensor: aliases external shared memory (D-03).
// T data[] is a zero-length array — lowers to ptr addrspace(3) in Phase 6.
// operator() takes logical indices, flattens via Shape strides, evaluates
// SharedLinearLayout to get byte offset, and returns T& (read+write, D-04).
template<typename T, typename TShape, typename TLayout>
struct SharedTensor {
    uint32_t __shared_memory_base;  // sentinel: satisfies C++ struct-inhabitant rule
                                 // for flexible array members (required by nvcc).
                                 // This struct is never allocated — it solely
                                 // aliases external shared memory through data[].

    static constexpr uint32_t RANK = TShape::RANK;

    // Row-major flatten of logical indices into a flat element index:
    // flatIndex = sum(indices[k] * stride[k]), inner dim varies fastest.
    static constexpr uint32_t flattenCoords(const uint32_t (&idxs)[RANK]) {
        constexpr auto& dims = ShapeDims<TShape>::All;
        uint32_t flatIndex = 0;
        for (int d = 0; d < RANK; ++d) {
            uint32_t stride = 1;
            for (int k = d + 1; k < RANK; ++k)
                stride *= dims[k];
            flatIndex += idxs[d] * stride;
        }
        return flatIndex;
    }

    // Byte offset of the element at flatIndex:
    // dot(TLayout::evaluate(flatIndex), row-major byte strides).
    static constexpr uint32_t byteOffsetOf(uint32_t flatIndex) {
        constexpr auto& dims = ShapeDims<TShape>::All;
        auto logicalOffset = TLayout::evaluate(flatIndex, IntTuple<RANK>{});
        uint32_t byteOffset = 0;
        for (int d = 0; d < RANK; ++d) {
            uint32_t byteStride = sizeof(T);
            for (int k = d + 1; k < RANK; ++k)
                byteStride *= dims[k];
            byteOffset += logicalOffset.Dims[d] * byteStride;
        }
        return byteOffset;
    }

    // Debug contiguity check: the N_ELEMS elements starting at flatBase must
    // have consecutive byte offsets with stride sizeof(T).
    template<uint32_t N_ELEMS>
    static constexpr bool isContiguous(uint32_t flatBase) {
        uint32_t base = byteOffsetOf(flatBase);
        for (uint32_t k = 1; k < N_ELEMS; ++k)
            if (byteOffsetOf(flatBase + k) != base + k * sizeof(T))
                return false;
        return true;
    }

    // D-04: variadic operator() accepting RANK logical indices, returning T&.
    __device__ __forceinline__ shared_accessor<T> operator()(auto... indices) {
        static_assert(sizeof...(indices) == RANK, "number of indices must match tensor rank");
        uint32_t idxs[RANK] = {static_cast<uint32_t>(indices)...};
        return shared_accessor<T>{__shared_memory_base + byteOffsetOf(flattenCoords(idxs))};
    }

    // Vector access at flat element index flatBase.
    template<uint32_t N_ELEMS>
    __device__ __forceinline__ void loadVector(uint32_t flatBase, T (&out)[N_ELEMS]) const {
        constexpr uint32_t N_BYTES = N_ELEMS * sizeof(T);
        uint32_t base = byteOffsetOf(flatBase);
        if (base % N_BYTES != 0) __trap();  // misaligned vector access
        assert(isContiguous<N_ELEMS>(flatBase) && "loadVector: elements not contiguous");
        shared_load_vector<T, N_ELEMS>(__shared_memory_base + base, out);
    }

    template<uint32_t N_ELEMS>
    __device__ __forceinline__ void storeVector(uint32_t flatBase, const T (&in)[N_ELEMS]) {
        constexpr uint32_t N_BYTES = N_ELEMS * sizeof(T);
        uint32_t base = byteOffsetOf(flatBase);
        if (base % N_BYTES != 0) __trap();  // misaligned vector access
        assert(isContiguous<N_ELEMS>(flatBase) && "storeVector: elements not contiguous");
        shared_store_vector<T, N_ELEMS>(__shared_memory_base + base, in);
    }

    // Asynchronous global→shared copy of N_ELEMS elements (4/8/16 bytes).
    // gsrc must point to N_ELEMS consecutive elements in global memory.
    template<uint32_t N_ELEMS>
    __device__ __forceinline__ void cpAsyncFromGlobal(const T* gsrc, uint32_t flatBase) {
        constexpr uint32_t N_BYTES = N_ELEMS * sizeof(T);
        static_assert(N_BYTES == 4 || N_BYTES == 8 || N_BYTES == 16,
                      "cp.async supports 4, 8 or 16 bytes per thread");
        uint32_t base = byteOffsetOf(flatBase);
        if (base % N_BYTES != 0) __trap();  // misaligned cp.async destination
        assert(isContiguous<N_ELEMS>(flatBase) && "cpAsyncFromGlobal: elements not contiguous");
        if constexpr (N_BYTES == 16) {
            cp_async_cg_16(__shared_memory_base + base, gsrc);
        } else {
            cp_async_ca(__shared_memory_base + base, gsrc, N_BYTES);
        }
    }
};

// ldmatrix.sync.aligned.m8n8.x4.shared.b16 — loads four 8x8 b16 matrices
// (a 16x16 tile of 2-byte elements). Lane l supplies the row address:
//   matrix m = l / 8, row = l % 8; matrices are blocked as
//   m0=(rows 0-7, cols 0-7), m1=(rows 8-15, cols 0-7),
//   m2=(rows 0-7, cols 8-15), m3=(rows 8-15, cols 8-15).
// After the load, lane l's out[i] holds matrix i's row (l/4), u32 column
// (l%4) — i.e. two adjacent b16 elements, matching the mma fragment layout.
// fragBase is the logical coordinate of the tile's top-left element.
template<typename T, typename TShape, typename TLayout>
__device__ __forceinline__ void ldmatrix_x4_b16(
    const SharedTensor<T, TShape, TLayout>& shm,
    IntTuple<TShape::RANK> fragBase, uint32_t (&out)[4]) {
    static_assert(sizeof(T) == 2, "ldmatrix_x4_b16 requires 2-byte elements");
    static_assert(TShape::RANK == 2, "ldmatrix tile must be rank 2");
    uint32_t lane = threadIdx.x & 31;
    uint32_t mat = lane >> 3, row = lane & 7;
    uint32_t idxs[2] = {fragBase.Dims[0] + row + (mat & 1) * 8,
                        fragBase.Dims[1] + (mat >> 1) * 8};
    uint32_t addr = shm.__shared_memory_base +
                    SharedTensor<T, TShape, TLayout>::byteOffsetOf(
                        SharedTensor<T, TShape, TLayout>::flattenCoords(idxs));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(addr) : "memory");
}

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
    uint32_t tid = threadIdx.x;
    #pragma unroll TLayout::REG_SIZE
    for (uint32_t i = 0; i < TLayout::REG_SIZE; i++)
        shm(i * 32 + tid) = shm(i * 32 + tid) + val.data[i]; // users should be fully aware of every reads and writes to shared memory
}

template<typename T, typename TLayout>
__device__ void write_swizzled_2d(SharedTensor<T, Shape<32, 16>, TLayout>& shm) {
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 16; j++)
            shm(i, j) = static_cast<T>(i * 16 + j);
}

// named_barrier_rotate_1d: each thread stores its register to shared memory,
// synchronizes the whole CTA on the CALLER-PROVIDED named barrier (the
// callee never decides the synchronization scope), then reads back the slot
// written by the next thread — out[i] = in[(i+1) % N]. REG_SIZE == 1, so a
// thread's slot index is simply threadIdx.x.
template<typename T, uint32_t N, typename SharedTLayout, typename TLayout>
__device__ Tensor<T, Shape<N>, TLayout> named_barrier_rotate_1d(
    SharedTensor<T, Shape<N>, SharedTLayout>& shm,
    const Tensor<T, Shape<N>, TLayout>& vals,
    const NamedBarrier& bar)
{
    static_assert(TLayout::REG_SIZE == 1,
                  "rotate test assumes one register per thread");
    uint32_t tid = threadIdx.x;
    Tensor<T, Shape<N>, TLayout> out;
    shm(tid) = vals.data[0];
    bar.sync();
    out.data[0] = shm((tid + 1) % N);
    return out;
}

// ========================= END OF DEFINITIONS =============================

// write_vals_1d: shm(i*32+tid) = vals.data[i] — scalar stores at the
// element width (b16 for half, b8 for int8).
template<typename T, uint32_t N, typename SharedTLayout, typename TLayout>
__device__ void write_vals_1d(
    SharedTensor<T, Shape<N>, SharedTLayout>& shm,
    const Tensor<T, Shape<N>, TLayout>& vals)
{
    uint32_t tid = threadIdx.x;
    #pragma unroll TLayout::REG_SIZE
    for (uint32_t i = 0; i < TLayout::REG_SIZE; i++)
        shm(i * 32 + tid) = vals.data[i];
}

// scale_shared_1d: shm(i) = shm(i) * factor — load+store round-trip at the
// element width (exercises b8/b16 ld.shared/st.shared). F is deduced from
// the gl.call scalar arg (f32), so arithmetic goes through float.
template<typename T, uint32_t N, typename TLayout, typename F>
__device__ void scale_shared_1d(SharedTensor<T, Shape<N>, TLayout>& shm, F factor) {
    for (uint32_t i = 0; i < N; i++)
        shm(i) = static_cast<T>(static_cast<float>(shm(i)) * static_cast<float>(factor));
}

// vector_scale_1d: same as scale_shared_1d but through 16-byte vector
// accesses (loadVector/storeVector).
template<typename T, uint32_t N, typename TLayout, typename F>
__device__ void vector_scale_1d(SharedTensor<T, Shape<N>, TLayout>& shm, F factor) {
    constexpr uint32_t VEC = 16 / sizeof(T);
    static_assert(N % VEC == 0, "vector width must divide N");
    for (uint32_t base = 0; base < N; base += VEC) {
        T vals[VEC];
        shm.template loadVector<VEC>(base, vals);
        #pragma unroll
        for (uint32_t i = 0; i < VEC; i++)
            vals[i] = static_cast<T>(static_cast<float>(vals[i]) * static_cast<float>(factor));
        shm.template storeVector<VEC>(base, vals);
    }
}

// ldmatrix_dump_frag: reads the 16x16 fp16 tile via ldmatrix.x4 and dumps
// each lane's four u32 fragment registers to shmFrag(lane, i) for host-side
// verification against the ldmatrix fragment layout definition. TFrag is
// deduced (int32 on the gluon side — MLIR "i32" is signless).
template<typename T, typename TFrag, typename TLayoutA, typename TLayoutB>
__device__ void ldmatrix_dump_frag(
    SharedTensor<T, Shape<16, 16>, TLayoutA>& shmTile,
    SharedTensor<TFrag, Shape<32, 4>, TLayoutB>& shmFrag)
{
    uint32_t out[4];
    ldmatrix_x4_b16(shmTile, IntTuple<2>{0, 0}, out);
    uint32_t lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        shmFrag(lane, i) = static_cast<TFrag>(out[i]);
}

// cp_async_fill_1d: each thread cp.asyncs one 16-byte chunk from global
// memory into its slot of the shared tensor. gsrcAddr is the global source
// address passed as an i64 scalar (pointer-to-int on the gluon side).
template<typename T, uint32_t N, typename TLayout>
__device__ void cp_async_fill_1d(SharedTensor<T, Shape<N>, TLayout>& shm,
                                 unsigned long long gsrcAddr) {
    constexpr uint32_t VEC = 16 / sizeof(T);
    static_assert(N == 32 * VEC, "one 16-byte chunk per thread");
    uint32_t tid = threadIdx.x;
    const T* gsrc = reinterpret_cast<const T*>(gsrcAddr);
    shm.template cpAsyncFromGlobal<VEC>(gsrc + tid * VEC, tid * VEC);
    cp_async_commit();
    cp_async_wait<0>();
}

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

using TArg = TensorLayout::Layout<BasisGroup<2,5>{{0,1},{0,2},{0,4},{0,8},{0,16}},BasisGroup<2,5>{{1,0},{2,0},{4,0},{8,0},{16,0}},BasisGroup<2,0>{}>;
using TRes = TensorLayout::Layout<BasisGroup<1,0>{},BasisGroup<1,5>{{1},{2},{4},{8},{16}},BasisGroup<1,0>{}>;


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