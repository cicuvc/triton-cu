#ifndef TRITONGPU_ATTRDEFS
#define TRITONGPU_ATTRDEFS

include "triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td"

//===----------------------------------------------------------------------===//
// Shared Layout Encoding
//===----------------------------------------------------------------------===//

def SwizzledSharedEncodingAttr
    : TritonGPU_Attr<"SwizzledSharedEncoding", "swizzled_shared_encoding",
                     [SharedEncodingTrait, LayoutEncodingTrait,
                      DeclareLayoutEncodingMethods]> {
  let mnemonic = "swizzled_shared";

  let description = [{
An encoding for tensors whose elements may be simultaneously accessed by
different GPU threads in the programs, via shared memory. In other words,
for all indices i \in Z^d, \mathcal{L}(i) = {0, 1, ..., 32*num_warps - 1}.

In order to avoid shared memory bank conflicts, elements may be swizzled.
Here are some examples.  In all cases, the input tensor is [0, 1, ..., n-1].

1. Basic swizzling

  #ttg.swizzled_shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3],  // xor with 0
  [ 5,  4,  7,  6],  // xor with 1
  [10, 11,  8,  9],  // xor with 2
  [15, 14, 13, 12]   // xor with 3

Here elements of row r are xor'ed with r (or more properly, in[r][c] ->
out[r][c^r]).

2. Multiple rows per phase

  #ttg.swizzled_shared<{vec=1, perPhase=2, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 4,  5,  6,  7],
  [ 9,  8, 11, 10],  // phase 1 (xor with 1)
  [13, 12, 15, 14]

Elements of row r are xor'ed with r/2.  In other words, perPhase=2
means that pairs of 2 rows get the same swizzling.

3. Max-phase applied

  #ttg.swizzled_shared<{vec=1, perPhase=1, maxPhase=2, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 5,  4,  7,  6],  // phase 1 (xor with 1)
  [ 8,  9, 10, 11],  // phase 0
  [13, 12, 15, 14],  // phase 1
  [16, 17, 18, 19],  // ...
  [21, 20, 23, 22],
  [24, 25, 26, 27],
  [29, 28, 31, 30]

Elements of row r are xor'ed with (r/2) % 2.  In other words, maxPhase=m has the
effect of limiting the maximum value of the xor to m-1.

4. Max-phase and per-phase

  #ttg.swizzled_shared<{vec=1, perPhase=2, maxPhase=2, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 4,  5,  6,  7],  // phase 0
  [ 9,  8, 11, 10],  // phase 1 (xor with 1)
  [13, 12, 15, 14],  // phase 1
  [16, 17, 18, 19],  // phase 0
  [20, 21, 22, 23],  // phase 0
  [25, 24, 27, 26],  // phase 1
  [29, 28, 31, 30]]  // phase 1

Here the xor value (the "phase", I guess?) changes every perPhase rows, up to a
maximum value of maxPhase-1.  In other words, elements of row r are xor'ed with
(r/2) % 2.

5. Adding vec

  #ttg.swizzled_shared<{vec=2, perPhase=1, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3,  4,  5,  6,  7],
  [10, 11,  8,  9, 14, 15, 12, 13],
  [20, 21, 22, 23, 16, 17, 18, 19],
  [30, 31, 28, 29, 26, 27, 24, 25]

When vec=2, elements are swizzled in pairs of 2.  In other words, the element at
(r,c) has value

  ((c / 2) ^ r) * 2 + (c % 2).
  }];

  // swizzle info: vec, perPhase, maxPhase
  // order: the fastest-changing axis first
  let parameters = (
    ins
    "unsigned":$vec,
    "unsigned":$perPhase,
    "unsigned":$maxPhase,
    ArrayRefParameter<"unsigned">:$order,
    "CGAEncodingAttr":$CGALayout
  );

  let builders = [
    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CGAEncodingAttr":$CGALayout,
                     "unsigned":$typeWidthInBit), [{
        bool needTrans = false; // default value
        return get(context, dotOpEnc, shape, order, CGALayout, typeWidthInBit, needTrans);
    }]>,

    // TODO(jlebar): This should not be an overload of
    // SwizzledSharedEncodingAttr::get().  It's misleading, because it does a bunch of
    // nontrivial work based on the given dotOpEnc.
    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CGAEncodingAttr":$CGALayout,
                     "unsigned":$typeWidthInBit,
                     "bool":$needTrans), [{

        auto mmaEnc = mlir::dyn_cast<NvidiaMmaEncodingAttr>(dotOpEnc.getParent());

        if(!mmaEnc)
          return get(context, 1, 1, 1, order, CGALayout);

        // ---- begin Ampere & Hopper ----
        if (mmaEnc.isAmpere() || mmaEnc.isHopper()) {
          return get(context, dotOpEnc.getOpIdx(), dotOpEnc.getKWidth(), shape, order, CGALayout, typeWidthInBit, needTrans);
        }

        // ---- not implemented ----
        llvm_unreachable("unsupported swizzling for provided MMA version");
    }]>,

    // NVIDIA constructor!
    // TODO(lezcano): We should totally get rid of all these constructors...
    AttrBuilder<(ins "int":$opIdx,
                     "unsigned":$kWidth,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CGAEncodingAttr":$CGALayout,
                     "unsigned":$bitwidth,
                     "bool":$needTrans), [{
        int K =  getShapePerCTA(CGALayout.getCTASplitNum(), shape)[order[0]];
        // Elems necessary to cover all the banks divided by the inner dimension
        // This packs a few rows together for small K
        int perPhase = std::max<int>(1024 / (bitwidth * K), 1);

        // FP64 MMAv2 uses an m8n8k4 tile rather than the m16n8 tile used by
        // the other MMAv2 variants.
        int mmaStride = bitwidth == 64 ? 4 : 8;
        int vec = 4 * kWidth;
        // needsTrans is equiv. to flipping the opIdx
        if (needTrans)
          std::swap(vec, mmaStride);
        assert(opIdx == 0 || opIdx == 1);
        int rank = order.size();
        int kDim = opIdx == 0 ? rank-1 : rank-2;
        if (order[0] != kDim)
          std::swap(vec, mmaStride);
        // Count how many vec elements are needed to cover all the banks
        int maxPhase = std::max(std::min<int>(mmaStride, 1024 / (vec * bitwidth)), 1);
        // Account for the row packing from perPhase: mmaStride / perPhase
        maxPhase = std::max(maxPhase / perPhase, 1);
        return get(context, vec, perPhase, maxPhase, order, CGALayout);
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CGAEncodingAttr":$CGALayout,
                     "Type":$eltTy), [{
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      return get(context, dotOpEnc, shape, order, CGALayout, bitwidth);
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CGAEncodingAttr":$CGALayout,
                     "Type":$eltTy,
                     "bool":$needTrans), [{
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      return get(context, dotOpEnc, shape, order, CGALayout, bitwidth, needTrans);
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def PaddedSharedEncodingAttr
    : TritonGPU_Attr<"PaddedSharedEncoding", "padded_shared_encoding",
                     [SharedEncodingTrait, DeclareLayoutEncodingMethods]> {
  let mnemonic = "padded_shared";

  let description = [{
An encoding for tensors whose elements may be simultaneously accessed by
different GPU threads in the programs, via shared memory. In other words,
for all indices i \in Z^d, \mathcal{L}(i) = {0, 1, ..., 32*num_warps - 1}.
Compared to SwizzledSharedEncodingAttr, this encoding combines padding with
element reordering via linear transformation (e.g. row permutation) to avoid
shared memory bank conflicts.

Formally, given a layout:
    padded_shared<[<interval_0>:+<pad_0>, <interval_1>:+<pad_1>, ...]>
We insert a padding of `<pad_i>` elements after every `<interval_i>` elements.
Multi interval-padding pairs are supported for flexibility of multi tiered
padding schemes; they compose in an additive manner. So for a 1-D tensor element
at index i, the corresponding shared memory location index is
    i + \sum_{k} (i / interval_k) * pad_k = 1
`<interval_i>` and `<pad_i>` all need to be power of two.

Some concrete examples ignoring the linear component, using `eM` to mean tensor
elements and `pN` to mean padding:

1. Single interval-padding pair:

   #ttg.padded_shared<[2:+2], {...}>
   [e0, e1, p0, p1,
    e2, e3, p2, p3,
    ...]

2. Double interval-padding pairs:

   #ttg.padded_shared<[2:+1, 4:+2], {...}>
   [e0, e1, p0,
    e2, e3, p1, p2, p3,
    e4, e5, p4,
    e6, e7, p5, p6, p7,
    ...]

Furthermore this encoding allows for a linear remapping from the 1-D shared
memory offset to logical n-D tensor elements. The remapping is given in the form
of linear bases mapping from offset to [dim0, dim1...dimN-1].
See LinearLayout.h for more details how linear layouts are applied to remap
elements.
Some concrete examples using `xN` and `yN` to mean the logical n-D tensor elements
and `pN` to mean padding:

1. 1D Single interval-padding with strided elements

    #ttg.padded_shared<[2:+2] {offset = [[2], [1]], block = []}>
    [x0, x2, p0 p1,
     x1, x3, p2, p3
     ...]

2. 2D single interval-padding with rearranged rows.

    #ttg.padded_shared<[16:+1] {offset = [[0, 1], [0, 2], /*gap, stride by 2 rows*/[2, 0], [4, 0], [1, 0]]], block = []}>
    [
      x0y0, x0y1, x0y2, x0y3,
      x2y0, x2y1, x2y2, x2y3,
      x4y0, x4y1, x4y2, x4y3,
      x6y0, x6y1, x6y2, x6y3,
      p0,
      x1y0, x1y1, x1y2, x1y3,
      x3y0, x3y1, x3y2, x3y3,
      x5y0, x5y1, x5y2, x5y3,
      x7y0, x7y1, x7y2, x7y3,
      p1,
    ]

For identity mappings a short form based on order and shape is used to increase readability. The following two encodings are the same:

    #ttg.padded_shared<[2:+2] {order = [1, 0], shape = [16, 32]}>
    #ttg.padded_shared<[2:+2] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0]], block = []}>


  }];

  let parameters = (ins
      ArrayRefParameter<"unsigned">:$intervals,
      ArrayRefParameter<"unsigned">:$paddings,
      LinearLayoutParam:$linearComponent
  );

  let builders = [
      AttrBuilder<(ins "ArrayRef<std::pair<unsigned, unsigned>>":$intervalPads,
                       "LinearLayout":$linearComponent)>,

      // Builder to create an identity mapping as the linear component
      AttrBuilder<(ins "ArrayRef<std::pair<unsigned, unsigned>>":$intervalPads,
                       "ArrayRef<unsigned>":$order, "ArrayRef<int64_t>":$shape,
                       "CGAEncodingAttr":$cgaLayout)>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Returns the order of the dimensions `dimName` of the layout.
    // If more than one dimension is of size one, it uses defaultOrder to determine
    // the order of the dimensions of size one.
    SmallVector<unsigned> orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const;
    SmallVector<unsigned> getOrder() const;

    // Returns the bases of the dimensions `dimName` of the linear_component.
    // If skipBroadcast is false, we count a base zero
    SmallVector<unsigned> basesPerDim(StringAttr dimName,
                                      bool skipBroadcast = true) const;

    unsigned getMinInterval() const {
      return *llvm::min_element(getIntervals());
    }

    // Returns the total number of elements including padding given the input
    // tensor shape.
    int64_t getPaddedSize(ArrayRef<int64_t> shape) const;
  }];
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}


def PartitionedSharedEncodingAttr
    : TritonGPU_Attr<"PartitionedSharedEncoding", "partitioned_shared_encoding",
                     [SharedEncodingTrait, DeclareLayoutEncodingMethods,  LayoutEncodingTrait]> {
  let mnemonic = "partitioned_shared";

  let description = [{
    An encoding for tensors whose elements are partitioned across multiple
    separate shared memory allocations. This reduces shared memory partition
    conflicts by splitting a tensor along a specific dimension into separate
    allocations.

    Parameters:
    - numPartitions: Number of distinct memory partitions (and separate buffers).
      Buffers in different partitions MUST be placed in different physical
      shared memory slots.
    - numGroups: Number of groups. Each group contains numPartitions
      consecutive pieces of the tensor.
    - partitionDim: The dimension along which the tensor is partitioned.
    - partitionLayout: The shared memory layout used within each piece.

    The total number of logical pieces is numPartitions * numGroups.
    Pieces are organized as:
      [Group 0: pieces 0..numPartitions-1]
      [Group 1: pieces numPartitions..2*numPartitions-1]
      ...

    ## Memory Allocation

    The allocator creates numPartitions buffers (NOT numLogicalPieces buffers).
    Each buffer contains all pieces from all groups that belong to that partition,
    concatenated together. Buffer size = pieceSize * numGroups.

    For example, with numPartitions=2, numGroups=4 and
    partitionDim=0 on a [128, 32] tensor:
    - Total 8 logical pieces, each of size [16, 32] (pieceSize = 16*32 elements)
    - Piece layout: [0, 1, 2, 3, 4, 5, 6, 7]
    - Partitions:    0  1  0  1  0  1  0  1
    - Groups:       |Grp0||Grp1||Grp2||Grp3|

    Physical allocation (2 buffers, each containing 4 pieces):
    - Buffer 0 (Partition 0): [Piece0 | Piece2 | Piece4 | Piece6]
    - Buffer 1 (Partition 1): [Piece1 | Piece3 | Piece5 | Piece7]

    Physical allocation guarantee: Buffers in different partitions MUST reside
    in distinct physical shared memory partitions.
  }];

  let parameters = (ins
      "unsigned":$numPartitions,
      "unsigned":$numGroups,
      "unsigned":$partitionDim,
      "SharedEncodingTrait":$partitionLayout
  );

  let extraClassDeclaration = [{
    /// Returns the total number of logical pieces.
    unsigned getNumLogicalPieces() const {
      return getNumPartitions() * getNumGroups();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def SharedLinearEncodingAttr
    : TritonGPU_Attr<"SharedLinearEncoding", "shared_linear_encoding",
                     [SharedEncodingTrait, LayoutEncodingTrait,
                      DeclareLayoutEncodingMethods]> {
  let mnemonic = "shared_linear";

  let description = [{
    Linear shared encodings mirror LinearEncodingAttr but operate on shared
    memory layouts. The LinearLayout parameter captures how shared memory
    offsets (and optionally blocks) map to logical tensor indices.
  }];

  let parameters = (ins LinearLayoutParam:$linearLayout, "unsigned":$layoutAlignment);

  let extraClassDeclaration = [{
    SmallVector<unsigned> basesPerDim(StringAttr dimName,
                                      bool skipBroadcast = true) const;
    SmallVector<unsigned> orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const;

    SmallVector<unsigned> getOrder() const;

    unsigned getRank() const { return getLinearLayout().getNumOutDims(); }

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;

    int32_t getAlignment() const { return static_cast<int32_t>(getLayoutAlignment()); }
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def NVMMASharedEncodingAttr : TritonGPU_Attr<"NVMMASharedEncoding", "nvmma_shared_encoding",
                     [DeclareSharedEncodingMethods, LayoutEncodingTrait,
                      DeclareLayoutEncodingMethods]> {
  let mnemonic = "nvmma_shared";

  let description = [{
    Represent blocked shared memory matching MMAv3/MMAv5 shared memory input.
    This is meant to represent 2d tiled blocked layout.
    The full layout representation is described here:
    https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout
    When the memdesc has more than 2 dimensions the tiling is applied to 8 rows even if the first outer dimension is smaller than 8.
    In this case `transposed` means that the contiguous dimension is the most outer dimension of the memdesc.
  }];


  // fp4Padded: Indicates that this encoding represents a mixed-precision fp4 operand in MMAv5 scaled dot, which needs
  // to be in the special padded layout as described in https://docs.nvidia.com/cuda/parallel-thread-execution/#packing-format-used-for-matrix-a-and-b-by-kind-mxf8f6f4-in-shared-memory
  let parameters = (
    ins
    "unsigned":$swizzlingByteWidth,
    "bool":$transposed,
    "unsigned":$elementBitWidth,
    "bool":$fp4Padded,
    "CGAEncodingAttr":$CGALayout
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CGAEncodingAttr":$CGALayout,
                     "Type":$eltTy,
                     "bool": $fp4Padded), [{
        auto shapePerCTA = getShapePerCTA(CGALayout.getCTASplitNum(), shape);
        int32_t swizzlingByteWidth = 0;
        unsigned eleBitWidth = eltTy.getIntOrFloatBitWidth();
        int packingFactor = fp4Padded ? 2 : 1;

        // get proper shared memory swizzling mode from the contiguous dimension
        // size of the origin blocked layout.
        auto contigDimSizeInByte = shapePerCTA[order[0]] * packingFactor * eleBitWidth / 8;
        if (contigDimSizeInByte >= 128 && contigDimSizeInByte % 128 == 0) {
          swizzlingByteWidth = 128;
        } else if (contigDimSizeInByte >= 64 && contigDimSizeInByte % 64 == 0) {
          swizzlingByteWidth = 64;
        } else if (contigDimSizeInByte >= 32 && contigDimSizeInByte % 32 == 0) {
          swizzlingByteWidth = 32;
        } else {
          swizzlingByteWidth = 0;
        }
        int flattenOutterDim = 1;
        for (int i = 1; i < shapePerCTA.size(); i++) {
          flattenOutterDim *= shapePerCTA[order[i]];
        }
        if (shapePerCTA.size() < 2 || flattenOutterDim < 8) {
          swizzlingByteWidth = 0;
        }
        bool transposed = order.size() > 1 && order[0] == 0;
        return $_get(context, swizzlingByteWidth, transposed, eleBitWidth, fp4Padded, CGALayout);
    }]>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    int getPerPhase() const;
    int getMaxPhase() const;
    int getVec() const;
  }];
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}


class DistributedEncoding<string name, string attrMnemonic, list<Trait> traits = []>
  : TritonGPU_Attr<name, attrMnemonic,
                   !listconcat([DistributedEncodingTrait, LayoutEncodingTrait,
                                DeclareLayoutEncodingMethods],
                               traits)> {

  let description = [{
Distributed encodings have a layout function L that is entirely characterized
by a d-dimensional tensor T. Note that L doesn't need to have the same shape
(or even the same rank) as the tensor it is encoding.

The layout function \mathcal{L} of this layout is then defined, for an
index `i` \in Z^d, as follows:

\mathcal{L}(T)[i_d] = L[(i_d + k_d*T.shape[d]) % L.shape[d]] \forall k_d such as i_d + k_d*T.shape[d] < L.shape[d]

Intuitively, when the tensor dim size T.shape[d] is larger than the layout
dim size L.shape[d], on that particular dim, we distribute values from the
tensor to threads mapped in the layout in a "wrapped around" manner, with
each thread owning multiple values.

OTOH, when the tensor dim size T.shape[d] is smaller than the layout
dim size L.shape[d], on that particular dim, we distribute values from the
tensor to threads mapped in the layout in a "broadcasted" manner, with
each value owned by multiple threads.

For example, for a tensor/layout pair
T = [x  x  x  x  x  x  x  x]
    [x  x  x  x  x  x  x  x]
L = [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

Then the data of T would be distributed as follow between the 16 CUDA threads:
L(T) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},
         {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]
  }];

  code extraDistributedDeclaration  = extraBaseClassDeclaration # [{
    // Implemented in subclasses
    SmallVector<unsigned> getRepOrder() const;

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
  }];
}

//===----------------------------------------------------------------------===//
// Linear Layout Encodings (common base class)
//===----------------------------------------------------------------------===//

class LinearDistributedEncoding<string name, string attrMnemonic,
                                list<Trait> traits = []>
  : DistributedEncoding<name, attrMnemonic,
      !listconcat([LinearEncodingTrait, DeclareLinearEncodingMethods],
                  traits)> {

  let parameters = (ins LinearLayoutParam:$linearLayout);

  code extraLinearDistributedDeclaration = [{
    unsigned getRank() const { return getLinearLayout().getNumOutDims(); }
  }];
}

//===----------------------------------------------------------------------===//
// Linear Layout Encoding
//===----------------------------------------------------------------------===//

def LinearEncodingAttr
    : LinearDistributedEncoding<"LinearEncoding", "linear_encoding"> {
  let mnemonic = "linear";

  let description = [{
    An encoding backed by a LinearLayout with input dims
    {register, lane, warp, block} and output dims {dim0, dim1, ...}. See the
    docs in LinearLayout.h for the general definition of linear layouts;
    this attribute is a strictly more restricted form of that abstraction.

    Compared to GenericLinearEncodingAttr, this is more constrained
    LinearLayout-based encoding:
    - Every basis vector (register, lane, warp, and block) must be
      non-swizzled, i.e. have at most one non-zero output component (the
      basis either moves along a single output dim, or is a broadcast/zero
      basis).
    - After removing broadcast bases the layout must be bijective.

    GenericLinearEncodingAttr relaxes the above in two ways:
    - Warp bases MAY be swizzled (non-zero in multiple output dims).
      Register, lane, and block bases must still be non-swizzled.
    - The layout is only required to be surjective (not bijective) after
      removing broadcast bases.
  }];

  let extraClassDeclaration = extraLinearDistributedDeclaration # [{
    SmallVector<unsigned> getWarpsPerCTA() const;
    SmallVector<unsigned> getWarpOrder() const;
  }];

  let genVerifyDecl = 1;
  // Example of assembly format:
  // <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
  //   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
  //   warp = [[16, 0], [32, 0]],
  //   block = []}>
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Generic Linear Layout Encoding
//===----------------------------------------------------------------------===//

def GenericLinearEncodingAttr
    : LinearDistributedEncoding<"GenericLinearEncoding",
                                "generic_linear_encoding"> {
  let mnemonic = "generic_linear";

  let description = [{
    A linear layout encoding with relaxed constraints compared to LinearEncoding.
    Like LinearEncoding, it wraps a LinearLayout with input dims
    {register, lane, warp, block} and output dims {dim0, dim1, ...}.

    Unlike LinearEncoding:
    - Warp bases MAY be swizzled (non-zero in multiple output dims).
    - The layout is NOT required to be bijective after removing broadcast bases,
      but it's still required to be surjective.

    Register, lane, and block bases must still be non-swizzled (at most one
    non-zero output component per basis vector), which preserves existing
    within-thread, within-warp, and CGA lowering paths.
  }];

  let extraClassDeclaration = extraLinearDistributedDeclaration;

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Blocked Layout Encoding
//===----------------------------------------------------------------------===//

def BlockedEncodingAttr : DistributedEncoding<"BlockedEncoding", "blocked_encoding"> {
  let mnemonic = "blocked";

  let description = [{
An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout
used to promote memory coalescing in LoadInst and StoreInst.
It is characterized by three tuples -- thread tile size, warp tile size, and block tile size -- which
specify the amount of elements owned by each CUDA thread, warp and CTA respectively.

Example 1, a row-major coalesced layout may partition a 16x16 tensor over 2 warps (i.e. 64 threads) as follows:

[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

for

#ttg.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  blocked = {{0, 1}}
}>

Example 2, a row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) as follows:

[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                 ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                 ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
for

#ttg.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  blocked = {{0, 1}}
}>

Example 3, A row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) and
4 CTAs (taking 2x2 for example) as follows:

CTA [0,0]                                              CTA [0,1]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                    ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

CTA [1,0]                                              CTA [1,1]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                    ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
for

#ttg.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  blocked = {{0, 1}, {1, 0}}
}>
}];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerThread,
    ArrayRefParameter<"unsigned">:$threadsPerWarp,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    ArrayRefParameter<"unsigned">:$order, // the fastest-changing axis first

    // CGALayout is optional in the textual IR.  If omitted, we infer it to be a
    // CGA with a single CTA (i.e. the trivial map onto dim0..dimn-1)
    "CGAEncodingAttr":$CGALayout
  );
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps,
                     "unsigned":$numThreadsPerWarp,
                     "CGAEncodingAttr":$CGALayout), [{
      unsigned rank = sizePerThread.size();
      SmallVector<unsigned, 4> threadsPerWarp(rank);
      SmallVector<unsigned, 4> warpsPerCTA(rank);
      SmallVector<int64_t> shapePerCTA = getShapePerCTA(CGALayout.getCTASplitNum(), shape);

      unsigned remainingLanes = numThreadsPerWarp;
      unsigned remainingThreads = numWarps * numThreadsPerWarp;
      unsigned remainingWarps = numWarps;
      unsigned prevLanes = 1;
      unsigned prevWarps = 1;

      // starting from the contiguous dimension
      for (unsigned d = 0; d < rank - 1; ++d) {
        unsigned i = order[d];
        unsigned threadsPerCTA = std::clamp<unsigned>(remainingThreads, 1, std::max<unsigned>(1, shapePerCTA[i] / sizePerThread[i]));
        threadsPerWarp[i] = std::clamp<unsigned>(threadsPerCTA, 1, remainingLanes);
        warpsPerCTA[i] = std::clamp<unsigned>(threadsPerCTA / threadsPerWarp[i], 1, remainingWarps);
        remainingWarps /= warpsPerCTA[i];
        remainingLanes /= threadsPerWarp[i];
        remainingThreads /= threadsPerCTA;
        prevLanes *= threadsPerWarp[i];
        prevWarps *= warpsPerCTA[i];
      }

      // Expand the last dimension to fill the remaining lanes and warps
      threadsPerWarp[order[rank - 1]] = numThreadsPerWarp / prevLanes;
      warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

      return $_get(context, sizePerThread, threadsPerWarp, warpsPerCTA, order, CGALayout);
    }]>,

    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps,
                     "unsigned":$numThreadsPerWarp,
                     "unsigned":$numCTAs), [{
      unsigned rank = sizePerThread.size();
      SmallVector<unsigned, 4> CTAsPerCGA(rank);
      SmallVector<unsigned, 4> CTASplitNum(rank);
      ArrayRef<unsigned> CTAOrder = order;

      unsigned remainingCTAs = numCTAs;

      // starting from the most strided dimension
      for (int d = rank - 1; d >= 0; --d) {
        unsigned i = order[d];
        CTAsPerCGA[i] = std::clamp<unsigned>(remainingCTAs, 1, std::max<unsigned>(1, shape[i] / sizePerThread[i]));
        CTASplitNum[i] = CTAsPerCGA[i];
        remainingCTAs /= CTAsPerCGA[i];
      }

      CTAsPerCGA[rank - 1] *= remainingCTAs; // wrap at CTA level

      CGAEncodingAttr CGALayout = CGAEncodingAttr::fromSplitParams(context, CTAsPerCGA, CTASplitNum, CTAOrder);
      return get(context, shape, sizePerThread, order, numWarps, numThreadsPerWarp, CGALayout);
    }]>
  ];

  let extraClassDeclaration = extraDistributedDeclaration;

  let hasCustomAssemblyFormat = 1;
}

def NvidiaMmaEncodingAttr : DistributedEncoding<"NvidiaMmaEncoding", "nvidia_mma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "nvidia_mma";

  let description = [{
An encoding for tensors that have been produced by tensor cores.

It is characterized by two parameters:
- A 'versionMajor' which specifies the generation the tensor cores
  whose output is being partitioned:
  - 1 for first-gen tensor cores (Volta), and
  - 2 for second-gen tensor cores (Turing/Ampere).
- A 'versionMinor' which indicates the specific layout of a tensor core
  generation, e.g. for Volta, there might be multiple kinds of layouts
  annotated by 0,1,2 and so on.
- A `blockTileSize` to indicate how data should be partitioned between warps.

// -------------------------------- version = 1 --------------------------- //

For first-gen tensor cores, the implicit warpTileSize is [16, 16].
Note: the layout is different from the recommended in PTX ISA
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.884 section, FP32 accumulator).

For example, when versionMinor=1, the matrix L corresponding to
blockTileSize=[32,16] is:

                               warp 0
--------------------------------/\-------------------------------
[ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
[ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
[ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
[ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
[ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
[ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
[ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
[ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
[ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
[ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
[ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
[ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
[ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
[ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]
[ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
[ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]

                          warp 1 = warp0 + 32
--------------------------------/\-------------------------------
[ 32  32  34  34  40  40  42  42   32  32  34  34  40  40  42  42 ]
[ 33  33  35  35  41  41  43  43   33  33  35  35  41  41  43  43 ]
[ ............................................................... ]


// -------------------------------- version = 2 --------------------------- //

For second-gen tensor cores, the implicit warpTileSize is [16, 8].
Information about this layout can be found in the official PTX documentation
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.16816 section, FP32 accumulator).

For example, the matrix L corresponding to blockTileSize=[32,16] is:
                warp 0                          warp 2
-----------------/\-------------  ----------------/\-------------
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63

              warp 1                           warp 3
----------------/\-------------   ----------------/\-------------
[ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
[ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
[ ..............................  ...............................
[ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127
[ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
[ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
[ ..............................  ...............................
[ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127

}];

  let parameters = (
    ins
    "unsigned":$versionMajor,
    "unsigned":$versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    "CGAEncodingAttr":$CGALayout,
    ArrayRefParameter<"unsigned">:$instrShape
  );


  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool isVolta() const;
    bool isTuring() const;
    bool isAmpere() const;
    bool isHopper() const;

    SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> shape,
                                          int bitwidth, int kWidth,
                                          int opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
  }];

  let hasCustomAssemblyFormat = 1;
}

def SliceEncodingAttr : DistributedEncoding<"SliceEncoding", "slice_encoding"> {
  let mnemonic = "slice";

  let description = [{
    Given a `parent` layout and a `dim`, squeezes the given `dim` in the `parent`
    layout and distributes values in a tensor T according to the new layout.

    For example, given

    T = [x  x  x  x  x  x  x  x]
    L_parent = [0  1  2  3 ]
               [4  5  6  7 ]
               [8  9  10 11]
               [12 13 14 15] (with 16 CUDA threads)

    With dim = 0, squeezing out dim 0, we have
    L = [{0,4,8,12},  {1,5,9,13}, {2,6,10,14},  {3,7,11,15} ]

    Then the data of T would be distributed as follow between the 16 CUDA threads:
    L(T) = [ {0,4,8,12} , {1,5,9,13} , ... {3,7,11,15}, {0,4,8,12} , ..., {3,7,11,15} ]

    With dim = 1, squeezing out dim 1, we have
    L = [ {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15} ]

    Then the data of T would be distributed as follow between the 16 CUDA threads:
    L = [ {0,1,2,3}, {4,5,6,7}, ..., {12,13,14,15}, {0,1,2,3}, ..., {12,13,14,15} ]

    This is useful for constructing the inverse layout of an expand_dims operation
    during some optimization passes.
  }];

  let parameters = (
    ins
    "unsigned":$dim,
    "DistributedEncodingTrait":$parent
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    template<class T>
    SmallVector<T> paddedShape(ArrayRef<T> shape) const;

    unsigned getRank() const {
      return mlir::cast<LayoutEncodingTrait>(getParent()).getRank() - 1;
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def DotOperandEncodingAttr : DistributedEncoding<"DotOperandEncoding", "dot_operand_encoding"> {
  let mnemonic = "dot_op";

  let description = [{
In the TritonGPU dialect, given `d = tt.dot a, b, c` tt.dot's operands a and b
must be of DotOperandEncodingAttr layout, if the dot is MMA v1 or v2 (i.e.
pre-Hopper).  For MMA v3, the operands are *almost always* in a regular shared
encoding, but sometimes the LHS is also a dot-operand encoding.

a's opIdx is 0, b's opIdx is 1.

The parent field is the layout of d.

kWidth defines number of consecutive elements stored by one thread along k dimension.
Some layouts do not use this parameter, either because they have a fixed number of
elements along the K dim, or they use all elements of the tensor along the K dim.

# WGMMA Notes
We require kWidth to be provided for Hopper because the dtype at loading might be
different from the dtype at WGMMA, due to casting. The kWidth is determined by the
dtype at WGMMA.

The encoded tensor consists of operand A for possibly multiple wgmma instructions.
For each wgmma, each warp in a warp group feeds a single "warp matrix"
Each warp matrix consists of 2x2 "quads".
Each thread holds several elements in each quad. Right before a wgmma,
the sum of bitwidth of
the elements in each quad should add up to 32.

These values are stored unrolled in `elements`.
The ordering of dimensions is as follows by convention:
batch (only 1 batch for Hopper currently)
matM (m-index of the "warp matrix")
matK (k-index of the "warp matrix")
quadK (k-index of the "quad" in the core matrix)
quadM (m-index of the "quad" in the core matrix)
vecIdx (index of the element in the quad; this is always along the k-dim)
  }];

  let parameters = (
    ins
    "unsigned":$opIdx,
    "Attribute":$parent,
    DefaultValuedParameter<"unsigned", "0">:$kWidth
  );

  let builders = [
    AttrBuilder<(ins "unsigned":$opIdx,
                     "Attribute":$parent,
                     "Type":$eltTy), [{
      NvidiaMmaEncodingAttr parentAttr = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent);
      if (!parentAttr || (!parentAttr.isAmpere() && !parentAttr.isHopper()))
        return $_get(context, opIdx, parent, 0);
      // For MMAV2 and V3
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      unsigned kWidth = std::max(32 / bitwidth, 1u);
      return $_get(context, opIdx, parent, kWidth);
    }]>
  ];

  let assemblyFormat = "`<` `{` struct(params) `}` `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = extraDistributedDeclaration;
}

def TTG_SharedMemorySpace : AttrDef<TritonGPU_Dialect, "SharedMemorySpace"> {
  let mnemonic = "shared_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to shared memory.
  }];
}

#endif
