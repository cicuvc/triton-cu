#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
std::ostream &operator<<(std::ostream &os, StringAttr str) {
  os << str.str();
  return os;
}
} // namespace mlir

using namespace mlir::triton::nvidia_gpu;
namespace mlir::triton::gpu {
namespace {

class LinearLayoutConversionsTest : public ::testing::Test {
public:
  void SetUp() {
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctx.getOrLoadDialect<TritonNvidiaGPUDialect>();
  }

  BlockedEncodingAttr blocked(ArrayRef<unsigned> spt, ArrayRef<unsigned> tpw,
                              ArrayRef<unsigned> wpb, ArrayRef<unsigned> cpg,
                              ArrayRef<unsigned> cSplit, ArrayRef<unsigned> ord,
                              ArrayRef<unsigned> cOrd) {
    return BlockedEncodingAttr::get(
        &ctx, spt, tpw, wpb, ord,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd));
  }

  NvidiaMmaEncodingAttr mma(unsigned versionMaj, unsigned versionMin,
                            ArrayRef<unsigned> instrShape,
                            ArrayRef<unsigned> wbp, ArrayRef<unsigned> cpg,
                            ArrayRef<unsigned> cSplit,
                            ArrayRef<unsigned> cOrd) {
    return NvidiaMmaEncodingAttr::get(
        &ctx, versionMaj, versionMin, wbp,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd), instrShape);
  }

  NvidiaMmaEncodingAttr mma(unsigned versionMaj, unsigned versionMin,
                            ArrayRef<unsigned> instrShape,
                            ArrayRef<unsigned> numWarps) {
    auto cgaLayout = CGAEncodingAttr::get1CTALayout(&ctx, numWarps.size());
    return NvidiaMmaEncodingAttr::get(&ctx, versionMaj, versionMin, numWarps,
                                      std::move(cgaLayout), instrShape);
  }

  DotOperandEncodingAttr dot(Attribute parent, int idx, int kWidth) {
    return DotOperandEncodingAttr::get(&ctx, idx, parent, /*kWidth=*/kWidth);
  }


  SliceEncodingAttr slice(DistributedEncodingTrait parent, int dim) {
    return SliceEncodingAttr::get(&ctx, dim, parent);
  }

  SwizzledSharedEncodingAttr shared(unsigned vec, unsigned perPhase,
                                    unsigned maxPhase, ArrayRef<unsigned> cpg,
                                    ArrayRef<unsigned> cSplit,
                                    ArrayRef<unsigned> ord,
                                    ArrayRef<unsigned> cOrd) {
    return SwizzledSharedEncodingAttr::get(
        &ctx, vec, perPhase, maxPhase, ord,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd));
  }

  NVMMASharedEncodingAttr
  nvmmaShared(unsigned swizzleSizeInBytes, bool transposed,
              unsigned elementBitWidth, ArrayRef<unsigned> cpg,
              ArrayRef<unsigned> cSplit, ArrayRef<unsigned> ord,
              ArrayRef<unsigned> cOrd, bool fp4Padded = false) {
    return NVMMASharedEncodingAttr::get(
        &ctx, swizzleSizeInBytes, transposed, elementBitWidth, fp4Padded,
        CGAEncodingAttr::fromSplitParams(&ctx, cpg, cSplit, cOrd));
  }


  TensorMemoryEncodingAttr tmem(unsigned blockM, unsigned blockN,
                                CGAEncodingAttr cgaLayout) {
    return TensorMemoryEncodingAttr::get(&ctx, blockM, blockN, 1, cgaLayout);
  }

  TensorMemoryEncodingAttr tmem(unsigned blockM, unsigned blockN) {
    // TODO Test colStride > 1
    return tmem(blockM, blockN, CGAEncodingAttr::get1CTALayout(&ctx, 2));
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LinearLayoutConversionsTest, SimpleBlocked) {
  auto layout =
      toLinearLayout({16}, blocked({1}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_THAT(layout, LinearLayout(
                          {
                              {S("register"), {}},
                              {S("lane"), {{1}, {2}}},
                              {S("warp"), {{4}, {8}}},
                              {S("block"), {}},
                          },
                          {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, CTADuplication) {
  auto layout = toLinearLayout(
      {32}, blocked({1}, {4}, {4}, /*cpg=*/{4}, /*cSplit=*/{2}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {}},
                            {S("lane"), {{1}, {2}}},
                            {S("warp"), {{4}, {8}}},
                            {S("block"), {{16}, {0}}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, CTABroadcast) {
  auto layout =
      toLinearLayout({64, 128}, blocked({8, 1}, {8, 4}, {1, 4}, {1, 2}, {1, 2},
                                        {0, 1}, {1, 0}));
  EXPECT_EQ(
      layout,
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 16}, {0, 32}}},
                    {S("lane"), {{8, 0}, {16, 0}, {32, 0}, {0, 1}, {0, 2}}},
                    {S("warp"), {{0, 4}, {0, 8}}},
                    {S("block"), {{0, 64}}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, ShapeLargerThanLayout) {
  // The layout is 16 elements, but the shape is 128, so it's repeated 128/16 =
  // 8 times.
  auto layout =
      toLinearLayout({128}, blocked({1}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{16}, {32}, {64}}},
                            {S("lane"), {{1}, {2}}},
                            {S("warp"), {{4}, {8}}},
                            {S("block"), {}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, ShapeLargerThanLayout2DDegenerate) {
  auto layout = toLinearLayout({128, 1}, blocked({1, 1}, {4, 1}, {4, 1}, {1, 1},
                                                 {1, 1}, {0, 1}, {1, 0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{16, 0}, {32, 0}, {64, 0}}},
                            {S("lane"), {{1, 0}, {2, 0}}},
                            {S("warp"), {{4, 0}, {8, 0}}},
                            {S("block"), {}},
                        },
                        {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, CanonicalScaleSmemLayout) {
  LinearLayout layout = getScaleSmemLayoutForTMEMCopy(
      &ctx, /*shape=*/{256, 16}, CGAEncodingAttr::get1CTALayout(&ctx, 2));
  LinearLayout expected = LinearLayout({{S("offset"),
                                         {{0, 1},
                                          {0, 2},
                                          {32, 0},
                                          {64, 0},
                                          {1, 0},
                                          {2, 0},
                                          {4, 0},
                                          {8, 0},
                                          {16, 0},
                                          {0, 4},
                                          {0, 8},
                                          {128, 0}}},
                                        {S("block"), {}}},
                                       {{S("dim0"), 256}, {S("dim1"), 16}},
                                       /*requireSurjective=*/true);
  EXPECT_EQ(layout, expected);
}

TEST_F(LinearLayoutConversionsTest, ShapeSmallerThanLayout) {
  // The shape is 8 elements, but the layout is 4*4*4 = 64 elems.  Therefore the
  // log2(64/8) = 3 most major bases are 0.
  auto layout = toLinearLayout({8}, blocked({4}, {4}, {4}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{1}, {2}}},
                            {S("lane"), {{4}, {0}}},
                            {S("warp"), {{0}, {0}}},
                            {S("block"), {}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, ReversedOrder) {
  auto layout = toLinearLayout({1, 64}, blocked({1, 1}, {32, 1}, {1, 8}, {1, 1},
                                                {1, 1}, {0, 1}, {1, 0}));
  EXPECT_EQ(layout,
            LinearLayout(
                {
                    {S("register"), {{0, 8}, {0, 16}, {0, 32}}},
                    {S("lane"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}},
                    {S("warp"), {{0, 1}, {0, 2}, {0, 4}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, ReplicateInRegisterDim) {
  auto layout =
      toLinearLayout({32}, blocked({2}, {4}, {1}, {1}, {1}, {0}, {0}));
  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"), {{1}, {8}, {16}}},
                            {S("lane"), {{2}, {4}}},
                            {S("warp"), {}},
                            {S("block"), {}},
                        },
                        {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, OneDimTooLargeAnotherTooSmall) {
  auto blockedLayout =
      blocked({1, 4}, {8, 4}, {4, 1}, {2, 2}, {2, 1}, {1, 0}, {1, 0});
  auto ll = toLinearLayout({128, 16}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {{0, 1}, {0, 2}, {32, 0}}},
                        {S("lane"), {{0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 0}}},
                        {S("warp"), {{8, 0}, {16, 0}}},
                        {S("block"), {{0, 0}, {64, 0}}},
                    },
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, RepeatInCTGDimFirst) {
  // We have a 4-element shape and an 8-element layout (4 elems per CTA).  So
  // the layout will map two inputs to each output.  The question is, which two
  // inputs?  The answer is, we split between CTAs first, so the two CTAs have
  // distinct elements.
  auto blockedLayout = blocked({1}, {1}, {4}, {2}, {2}, {0}, {0});
  auto ll = toLinearLayout({4}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {}},
                        {S("lane"), {}},
                        {S("warp"), {{1}, {0}}},
                        {S("block"), {{2}}},
                    },
                    {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, SmallerThanCGALayout) {
  auto blockedLayout = blocked({1}, {1}, {1}, {4}, {4}, {0}, {0});
  auto ll = toLinearLayout({2}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {}},
                        {S("lane"), {}},
                        {S("warp"), {}},
                        {S("block"), {{1}, {0}}},
                    },
                    {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, Skinny) {
  auto blockedLayout =
      blocked({8, 1}, {8, 4}, {1, 4}, {1, 2}, {1, 2}, {0, 1}, {0, 1});
  auto ll = toLinearLayout({64, 1}, blockedLayout);
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                        {S("lane"), {{8, 0}, {16, 0}, {32, 0}, {0, 0}, {0, 0}}},
                        {S("warp"), {{0, 0}, {0, 0}}},
                        {S("block"), {{0, 0}}},
                    },
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedOrder) {
  auto ll = toLinearLayout({1024, 128}, blocked({2, 2}, {4, 8}, {2, 2}, {2, 2},
                                                {2, 2}, {1, 0}, {1, 0}));
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"),
                         {
                             {0, 1},
                             {1, 0},
                             {0, 32},
                             {16, 0},
                             {32, 0},
                             {64, 0},
                             {128, 0},
                             {256, 0},
                         }},
                        {S("lane"), {{0, 2}, {0, 4}, {0, 8}, {2, 0}, {4, 0}}},
                        {S("warp"), {{0, 16}, {8, 0}}},
                        {S("block"), {{0, 64}, {512, 0}}},
                    },
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, Blocked4D) {
  auto ll = toLinearLayout({2, 1, 1, 1},
                           blocked({1, 1, 1, 4}, {2, 1, 1, 16}, {1, 2, 4, 1},
                                   {1, 1, 1, 1}, {1, 1, 1, 1}, {3, 0, 1, 2},
                                   {3, 2, 1, 0}));
  EXPECT_EQ(ll, LinearLayout(
                    {
                        {S("register"), {{0, 0, 0, 0}, {0, 0, 0, 0}}},
                        {S("lane"),
                         {{0, 0, 0, 0},
                          {0, 0, 0, 0},
                          {0, 0, 0, 0},
                          {0, 0, 0, 0},
                          {1, 0, 0, 0}}},
                        {S("warp"), {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
                        {S("block"), {}},
                    },
                    {S("dim0"), S("dim1"), S("dim2"), S("dim3")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDotOperandLhs) {
  auto parent = blocked(/*size*/ {2, 4}, /*threads*/ {8, 4}, /*warps*/ {2, 4},
                        /*ctas*/ {1, 1}, /*splits*/ {1, 1}, /*order*/ {1, 0},
                        /*cta order*/ {1, 0});
  auto dotOperand = dot(parent, /*idx*/ 0, /*kWidth*/ 0);
  EXPECT_EQ(
      toLinearLayout({32, 16}, dotOperand),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                    {S("lane"), {{0, 0}, {0, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDot3dOperandLhs) {
  auto parent =
      blocked(/*size*/ {2, 2, 4}, /*threads*/ {2, 4, 4}, /*warps*/ {2, 2, 2},
              /*ctas*/ {1, 1, 1}, /*splits*/ {1, 1, 1}, /*order*/ {2, 1, 0},
              /*cta order*/ {2, 1, 0});
  auto dotOperand = dot(parent, /*idx*/ 0, /*kWidth*/ 0);
  EXPECT_EQ(
      toLinearLayout({16, 32, 4}, dotOperand),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 1, 0},
             {1, 0, 0},
             {0, 16, 0},
             {8, 0, 0}}},
           {S("lane"), {{0, 0, 0}, {0, 0, 0}, {0, 2, 0}, {0, 4, 0}, {2, 0, 0}}},
           {S("warp"), {{0, 0, 0}, {0, 8, 0}, {4, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDotOperandRhs) {
  auto parent = blocked(/*size*/ {2, 4}, /*threads*/ {8, 4}, /*warps*/ {2, 4},
                        /*ctas*/ {1, 1}, /*splits*/ {1, 1}, /*order*/ {1, 0},
                        /*cta order*/ {1, 0});
  auto dotOperand = dot(parent, /*idx*/ 1, /*kWidth*/ 0);
  EXPECT_EQ(toLinearLayout({16, 64}, dotOperand),
            LinearLayout({{S("register"),
                           {{0, 1}, {0, 2}, {1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                          {S("lane"), {{0, 4}, {0, 8}, {0, 0}, {0, 0}, {0, 0}}},
                          {S("warp"), {{0, 16}, {0, 32}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, BlockedDot3dOperandRhs) {
  auto parent =
      blocked(/*size*/ {2, 2, 4}, /*threads*/ {2, 4, 4}, /*warps*/ {2, 2, 2},
              /*ctas*/ {1, 1, 1}, /*splits*/ {1, 1, 1}, /*order*/ {2, 1, 0},
              /*cta order*/ {2, 1, 0});
  auto dotOperand = dot(parent, /*idx*/ 1, /*kWidth*/ 0);
  EXPECT_EQ(
      toLinearLayout({16, 4, 64}, dotOperand),
      LinearLayout(
          {{S("register"),
            {{0, 0, 1},
             {0, 0, 2},
             {0, 1, 0},
             {0, 2, 0},
             {1, 0, 0},
             {0, 0, 32},
             {8, 0, 0}}},
           {S("lane"), {{0, 0, 4}, {0, 0, 8}, {0, 0, 0}, {0, 0, 0}, {2, 0, 0}}},
           {S("warp"), {{0, 0, 16}, {0, 0, 0}, {4, 0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_16x16) {
  EXPECT_EQ(toLinearLayout({16, 16},
                           mma(2, 0, {16, 8}, {1, 1}, {1, 1}, {1, 1}, {0, 1})),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_32x32) {
  EXPECT_EQ(toLinearLayout({32, 32},
                           mma(2, 0, {16, 8}, {1, 1}, {1, 1}, {1, 1}, {0, 1})),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {16, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_ExtendDim2) {
  EXPECT_EQ(toLinearLayout({16, 128},
                           mma(2, 0, {16, 8}, {1, 1}, {1, 1}, {1, 1}, {0, 1})),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {0, 32}, {0, 64}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_Cga) {
  EXPECT_EQ(
      toLinearLayout({64, 128, 128}, mma(2, 0, {1, 16, 8}, {16, 1, 1},
                                         {4, 2, 2}, {4, 2, 1}, {2, 1, 0})),
      LinearLayout(
          {
              {S("register"),
               {
                   {0, 0, 1},
                   {0, 8, 0},
                   {0, 0, 8},
                   {0, 0, 16},
                   {0, 0, 32},
                   {0, 0, 64},
                   {0, 16, 0},
                   {0, 32, 0},
               }},
              {S("lane"),
               {{0, 0, 2}, {0, 0, 4}, {0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
              {S("warp"), {{1, 0, 0}, {2, 0, 0}, {4, 0, 0}, {8, 0, 0}}},
              {S("block"), {{0, 0, 0}, {0, 64, 0}, {16, 0, 0}, {32, 0, 0}}},
          },
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv2_Small3D) {
  EXPECT_EQ(toLinearLayout({1, 128, 128}, mma(2, 0, {1, 16, 8}, {16, 1, 1},
                                              {4, 2, 2}, {4, 2, 1}, {2, 1, 0})),
            LinearLayout(
                {
                    {S("register"),
                     {
                         {0, 0, 1},
                         {0, 8, 0},
                         {0, 0, 8},
                         {0, 0, 16},
                         {0, 0, 32},
                         {0, 0, 64},
                         {0, 16, 0},
                         {0, 32, 0},
                     }},
                    {S("lane"),
                     {{0, 0, 2}, {0, 0, 4}, {0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
                    {S("block"), {{0, 0, 0}, {0, 64, 0}, {0, 0, 0}, {0, 0, 0}}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_64x16) {
  SmallVector<SmallVector<unsigned>, 2> instrShapes = {{16, 16, 8}, {16, 8, 8}};
  for (auto instrShape : instrShapes) {
    SCOPED_TRACE(triton::join(instrShape, ","));
    EXPECT_EQ(toLinearLayout({64, 16}, mma(3, 0, instrShape, {4, 1}, {1, 1},
                                           {1, 1}, {1, 0})),
              LinearLayout(
                  {
                      {S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                      {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                      {S("warp"), {{16, 0}, {32, 0}}},
                      {S("block"), {}},
                  },
                  {S("dim0"), S("dim1")}));
  }
}

TEST_F(LinearLayoutConversionsTest, MMAv3_128x16) {
  EXPECT_EQ(toLinearLayout({128, 16}, mma(3, 0, {16, 16, 8}, {4, 1}, {1, 1},
                                          {1, 1}, {1, 0})),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {64, 0}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_1024x1024) {
  EXPECT_EQ(toLinearLayout({1024, 1024}, mma(3, 0, {16, 16, 8}, {4, 1}, {1, 1},
                                             {1, 1}, {1, 0})),
            LinearLayout({{S("register"),
                           {{0, 1},
                            {8, 0},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {0, 64},
                            {0, 128},
                            {0, 256},
                            {0, 512},
                            {64, 0},
                            {128, 0},
                            {256, 0},
                            {512, 0}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_4x2Warps) {
  auto legacy = mma(3, 0, {16, 32, 16}, {4, 2}, {1, 1}, {1, 1}, {1, 0});

  EXPECT_EQ(toLinearLayout({64, 32}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 64}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 32}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({128, 64}, legacy),
      LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {64, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}, {0, 32}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({256, 64}, legacy),
      LinearLayout({{S("register"),
                     {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {64, 0}, {128, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}, {0, 32}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv3_4x4Warps) {
  auto legacy = mma(3, 0, {16, 16, 8}, {4, 4}, {1, 1}, {1, 1}, {1, 0});

  EXPECT_EQ(toLinearLayout({16, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {0, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 16}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}, {64, 0}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 0}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({32, 32}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {0, 0}, {0, 16}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 32}, legacy),
            LinearLayout({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                          {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                          {S("warp"), {{16, 0}, {32, 0}, {0, 16}, {0, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_tile_kwidth8) {
  auto parent = mma(2, 0, {16, 8}, {1, 1});
  EXPECT_EQ(toLinearLayout({16, 64}, dot(parent, 0, 8)),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}}},
                    {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 8}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_large_warp4_kwidth8) {
  auto parent = mma(2, 0, {16, 8}, {4, 1});
  EXPECT_EQ(
      toLinearLayout({128, 128}, dot(parent, 0, 8)),
      LinearLayout(
          {
              {S("register"),
               {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {64, 0}}},
              {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
              {S("warp"), {{16, 0}, {32, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({128, 64}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{1, 0},
                      {2, 0},
                      {4, 0},
                      {32, 0},
                      {64, 0},
                      {0, 8},
                      {0, 16},
                      {0, 32}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {
                        S("warp"),
                        {{0, 0}, {0, 0}},
                    },
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{1, 0},
                      {2, 0},
                      {4, 0},
                      {32, 0},
                      {0, 8},
                      {0, 16},
                      {0, 32},
                      {0, 64}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {
                        S("warp"),
                        {{0, 0}, {0, 0}},
                    },
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_3D) {
  // We implement one that exercises all the paths
  auto parent = mma(2, 0, {1, 16, 8}, {2, 4, 2});
  EXPECT_EQ(toLinearLayout({16, 128, 128}, dot(parent, 0, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 0, 1},
                      {0, 0, 2},
                      {0, 0, 4},
                      {0, 8, 0},
                      {0, 0, 32},
                      {0, 0, 64},
                      {0, 64, 0},
                      {2, 0, 0},
                      {4, 0, 0},
                      {8, 0, 0}}},
                    {S("lane"),
                     {{0, 0, 8}, {0, 0, 16}, {0, 1, 0}, {0, 2, 0}, {0, 4, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 16, 0}, {0, 32, 0}, {1, 0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
  EXPECT_EQ(toLinearLayout({8, 128, 64}, dot(parent, 1, 8)),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1, 0},
                      {0, 2, 0},
                      {0, 4, 0},
                      {0, 32, 0},
                      {0, 64, 0},
                      {0, 0, 16},
                      {0, 0, 32},
                      {2, 0, 0},
                      {4, 0, 0}}},
                    {S("lane"),
                     {{0, 8, 0}, {0, 16, 0}, {0, 0, 1}, {0, 0, 2}, {0, 0, 4}}},
                    {
                        S("warp"),
                        {{0, 0, 8}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}},
                    },
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv3_warp4_kwidth2) {
  auto parent = mma(3, 0, {16, 16, 8}, {4, 1});
  auto dotOp = dot(parent, 0, 2);

  EXPECT_EQ(toLinearLayout({64, 16}, dotOp),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 16}, dotOp),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}, {64, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(toLinearLayout({128, 32}, dotOp),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {0, 8}, {0, 16}, {64, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv3_mixed_warp_kwidth4) {
  // Testing dot with MMAv3 encoding for opIdx = 0 and kWidth = 4
  auto parent = mma(3, 0, {16, 16, 8}, {4, 2});
  auto dotOp = dot(parent, 0, 4);

  EXPECT_EQ(toLinearLayout({128, 64}, dotOp),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1}, {0, 2}, {8, 0}, {0, 16}, {0, 32}, {64, 0}}},
                    {S("lane"), {{0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{16, 0}, {32, 0}, {0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, DotMMAv2_split_warp_kwidth8) {
  auto parent = mma(2, 0, {16, 8}, {2, 2});
  EXPECT_EQ(
      toLinearLayout({32, 64}, dot(parent, 0, 8)),
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}}},
                    {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({64, 16}, dot(parent, 1, 8)),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}}},
                    {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
                    {S("warp"), {{0, 8}, {0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
  EXPECT_EQ(toLinearLayout({64, 128}, dot(parent, 0, 8)),
            LinearLayout(
                {{S("register"),
                  {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {32, 0}}},
                 {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
                 {S("warp"), {{0, 0}, {16, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      toLinearLayout({128, 32}, dot(parent, 1, 8)),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {32, 0}, {64, 0}, {0, 16}}},
           {S("lane"), {{8, 0}, {16, 0}, {0, 1}, {0, 2}, {0, 4}}},
           {S("warp"), {{0, 8}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SliceDot) {
  // Slice layout with a DotOperand (MMAv2) as the parent.
  auto parentV2 = dot(mma(2, 0, {16, 8}, {1, 1}), /*opIdx=*/0, /*kWidth=*/8);
  auto sliceV2 = slice(parentV2, /*dim=*/1);

  EXPECT_EQ(toLinearLayout({16}, sliceV2),
            LinearLayout(
                {
                    {S("register"), {{8}}},
                    {S("lane"), {{0}, {0}, {1}, {2}, {4}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0")}));

  // Slice layout with a DotOperand (MMAv3) as the parent.
  auto parentV3 =
      dot(mma(3, 0, {16, 16, 8}, {4, 1}), /*opIdx=*/0, /*kWidth=*/2);
  auto sliceV3 = slice(parentV3, /*dim=*/0);

  EXPECT_EQ(toLinearLayout({16}, sliceV3),
            LinearLayout(
                {
                    {S("register"), {{1}, {8}}},
                    {S("lane"), {{2}, {4}, {0}, {0}, {0}}},
                    {S("warp"), {{0}, {0}}},
                    {S("block"), {}},
                },
                {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, SliceOfBlocked) {
  auto parent = blocked({2, 4}, {4, 2}, {2, 2}, {2, 2}, {2, 2}, {1, 0}, {1, 0});
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 0)),
            LinearLayout({{S("register"), {{1}, {2}, {16}, {32}}},
                          {S("lane"), {{4}, {0}, {0}}},
                          {S("warp"), {{8}, {0}}},
                          {S("block"), {{64}, {0}}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 1)),
            LinearLayout({{S("register"), {{1}, {16}, {32}}},
                          {S("lane"), {{0}, {2}, {4}}},
                          {S("warp"), {{0}, {8}}},
                          {S("block"), {{0}, {64}}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, SliceWithShape1) {
  auto parent = blocked({1, 4}, {8, 4}, {2, 2}, {1, 1}, {1, 1}, {0, 1}, {1, 0});
  EXPECT_EQ(toLinearLayout({1}, slice(parent, 0)),
            LinearLayout({{S("register"), {}},
                          {S("lane"), {{0}, {0}, {0}, {0}, {0}}},
                          {S("warp"), {{0}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, Slice4D) {
  auto parent = blocked({1, 1, 1, 4}, {2, 1, 1, 16}, {1, 2, 4, 1}, {1, 1, 1, 1},
                        {1, 1, 1, 1}, {3, 0, 1, 2}, {3, 2, 1, 0});
  EXPECT_EQ(toLinearLayout({2, 1, 1}, slice(parent, 3)),
            LinearLayout(
                {
                    {S("register"), {}},
                    {S("lane"),
                     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(LinearLayoutConversionsTest, SliceOfMmaV2) {
  auto parent = mma(2, 0, {16, 8}, {2, 2}, {1, 1}, {1, 1}, {0, 1});
  EXPECT_EQ(toLinearLayout({16}, slice(parent, 0)),
            LinearLayout({{S("register"), {{1}}},
                          {S("lane"), {{2}, {4}, {0}, {0}, {0}}},
                          {S("warp"), {{8}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 0)),
            LinearLayout({{S("register"), {{1}, {16}, {32}, {64}}},
                          {S("lane"), {{2}, {4}, {0}, {0}, {0}}},
                          {S("warp"), {{8}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({8}, slice(parent, 1)),
            LinearLayout({{S("register"), {}},
                          {S("lane"), {{0}, {0}, {1}, {2}, {4}}},
                          {S("warp"), {{0}, {0}}},
                          {S("block"), {}}},
                         {S("dim0")}));
  EXPECT_EQ(toLinearLayout({128}, slice(parent, 1)),
            LinearLayout({{S("register"), {{8}, {32}, {64}}},
                          {S("lane"), {{0}, {0}, {1}, {2}, {4}}},
                          {S("warp"), {{0}, {16}}},
                          {S("block"), {}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSimple1D) {
  EXPECT_EQ(toLinearLayout({1024}, shared(1, 1, 1, {1}, {1}, {0}, {0})),
            LinearLayout::identity1D(1024, S("offset"), S("dim0")) *
                LinearLayout::identity1D(1, S("block"), S("dim0")));
}

TEST_F(LinearLayoutConversionsTest, SharedSimple2D) {
  EXPECT_EQ(toLinearLayout({128, 128},
                           shared(1, 1, 1, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
            (LinearLayout::identity1D(128, S("offset"), S("dim1")) *
             LinearLayout::identity1D(128, S("offset"), S("dim0")) *
             LinearLayout::identity1D(1, S("block"), S("dim0")))
                .transposeOuts({S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSimple2D_Order01) {
  EXPECT_EQ(toLinearLayout({128, 128},
                           shared(1, 1, 1, {1, 1}, {1, 1}, {0, 1}, {1, 0})),
            LinearLayout::identity1D(128, S("offset"), S("dim0")) *
                LinearLayout::identity1D(128, S("offset"), S("dim1")) *
                LinearLayout::identity1D(1, S("block"), S("dim0")));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_MaxPhaseOnly) {
  EXPECT_EQ(
      toLinearLayout({32, 32}, shared(1, 1, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 8},
                      {0, 16},
                      {1, 1},
                      {2, 2},
                      {4, 0},
                      {8, 0},
                      {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_PerPhaseMaxPhase) {
  EXPECT_EQ(
      toLinearLayout({32, 32}, shared(1, 2, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 8},
                      {0, 16},
                      {1, 0},
                      {2, 1},
                      {4, 2},
                      {8, 0},
                      {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_Vec) {
  EXPECT_EQ(
      toLinearLayout({4, 8}, shared(2, 1, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"), {{0, 1}, {0, 2}, {0, 4}, {1, 2}, {2, 4}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_PerPhaseMaxPhaseVec) {
  EXPECT_EQ(
      toLinearLayout({32, 32}, shared(2, 2, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1},
                      {0, 2},
                      {0, 4},
                      {0, 8},
                      {0, 16},
                      {1, 0},
                      {2, 2},
                      {4, 4},
                      {8, 0},
                      {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled4D) {
  EXPECT_EQ(
      toLinearLayout({2, 4, 32, 32}, shared(2, 2, 4, {1, 1, 1, 1}, {1, 1, 1, 1},
                                            {3, 2, 1, 0}, {3, 2, 1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 0, 0, 1},
                      {0, 0, 0, 2},
                      {0, 0, 0, 4},
                      {0, 0, 0, 8},
                      {0, 0, 0, 16},
                      {0, 0, 1, 0},
                      {0, 0, 2, 2},
                      {0, 0, 4, 4},
                      {0, 0, 8, 0},
                      {0, 0, 16, 0},
                      {0, 1, 0, 0},
                      {0, 2, 0, 0},
                      {1, 0, 0, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1"), S("dim2"), S("dim3")}));
}

TEST_F(LinearLayoutConversionsTest, SharedSwizzled2D_Order01) {
  EXPECT_EQ(
      toLinearLayout({4, 8}, shared(1, 1, 4, {1, 1}, {1, 1}, {0, 1}, {0, 1})),
      LinearLayout({{S("offset"), {{1, 0}, {2, 0}, {1, 1}, {2, 2}, {0, 4}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x16_4_2) {
  EXPECT_EQ(
      toLinearLayout(
          {8, 16}, nvmmaShared(32, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 8}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_128x16_4_2) {
  EXPECT_EQ(toLinearLayout({128, 16}, nvmmaShared(32, false, 16, {1, 1}, {1, 1},
                                                  {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {1, 0},
                            {2, 0},
                            {4, 8},
                            {8, 0},
                            {16, 0},
                            {32, 0},
                            {64, 0}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x32_2_4) {
  EXPECT_EQ(
      toLinearLayout(
          {8, 32}, nvmmaShared(64, false, 16, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout(
          {{S("offset"),
            {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {1, 0}, {2, 8}, {4, 16}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x64_1_8) {
  EXPECT_EQ(toLinearLayout({8, 64}, nvmmaShared(128, false, 16, {1, 1}, {1, 1},
                                                {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {1, 8},
                            {2, 16},
                            {4, 32}}},
                          {S("block"), {}}},
                         {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_8x64_1_8_32b) {
  EXPECT_EQ(toLinearLayout({8, 64}, nvmmaShared(128, false, 32, {1, 1}, {1, 1},
                                                {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
                           {{0, 1},
                            {0, 2},
                            {0, 4},
                            {0, 8},
                            {0, 16},
                            {1, 4},
                            {2, 8},
                            {4, 16},
                            {0, 32}}},
                          {S("block"), {}}},
                         {{S("dim0"), 8}, {S("dim1"), 64}},
                         /*requireSurjective=*/false));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_128x128_1_8_128b_transposed) {
  EXPECT_EQ(toLinearLayout({128, 128}, nvmmaShared(128, true, 32, {1, 1},
                                                   {1, 1}, {1, 0}, {1, 0})),
            LinearLayout({{S("offset"),
                           {{1, 0},
                            {2, 0},
                            {4, 0},
                            {8, 0},
                            {16, 0},
                            {4, 1},
                            {8, 2},
                            {16, 4},
                            {0, 8},
                            {0, 16},
                            {0, 32},
                            {0, 64},
                            {32, 0},
                            {64, 0}}},
                          {S("block"), {}}},
                         {{S("dim0"), 128}, {S("dim1"), 128}},
                         /*requireSurjective=*/true));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_32x4x64_1_8_32b) {
  EXPECT_EQ(
      toLinearLayout({32, 4, 64}, nvmmaShared(64, false, 32, {1, 1, 1},
                                              {1, 1, 1}, {2, 1, 0}, {2, 1, 0})),
      LinearLayout({{S("offset"),
                     {{0, 0, 1},
                      {0, 0, 2},
                      {0, 0, 4},
                      {0, 0, 8},
                      {0, 1, 0},
                      {0, 2, 4},
                      {1, 0, 8},
                      {2, 0, 0},
                      {4, 0, 0},
                      {8, 0, 0},
                      {16, 0, 0},
                      {0, 0, 16},
                      {0, 0, 32}}},
                    {S("block"), {}}},
                   {{S("dim0"), 32}, {S("dim1"), 4}, {S("dim2"), 64}},
                   /*requireSurjective=*/true));
}

TEST_F(LinearLayoutConversionsTest, LeadingOffset_64x4x32_1_8_32b_transposed) {
  EXPECT_EQ(
      toLinearLayout({64, 4, 32}, nvmmaShared(64, true, 32, {1, 1, 1},
                                              {1, 1, 1}, {2, 1, 0}, {2, 1, 0})),
      LinearLayout({{S("offset"),
                     {{1, 0, 0},
                      {2, 0, 0},
                      {4, 0, 0},
                      {8, 0, 0},
                      {0, 0, 4},
                      {4, 0, 8},
                      {8, 0, 16},
                      {0, 1, 0},
                      {0, 2, 0},
                      {0, 0, 1},
                      {0, 0, 2},
                      {16, 0, 0},
                      {32, 0, 0}}},
                    {S("block"), {}}},
                   {{S("dim0"), 64}, {S("dim1"), 4}, {S("dim2"), 32}},
                   /*requireSurjective=*/true));
}

TEST_F(LinearLayoutConversionsTest, Shared1DSwizzle) {
  EXPECT_EQ(
      toLinearLayout({64, 1}, shared(2, 2, 4, {1, 1}, {1, 1}, {1, 0}, {1, 0})),
      LinearLayout::identity1D(64, S("offset"), S("dim0")) *
          LinearLayout::identity1D(1, S("offset"), S("dim1")) *
          LinearLayout::identity1D(1, S("block"), S("dim0")));
}


TEST_F(LinearLayoutConversionsTest, ChooseShmemLayout) {
  LinearLayout ll = LinearLayout({{S("register"), {{1}, {2}, {2}, {8}}},
                                  {S("lane"), {{8}, {4}, {1}}},
                                  {S("warp"), {{16}, {32}, {0}}},
                                  {S("block"), {}}},
                                 {S("dim0")});
  EXPECT_EQ(chooseShemLayoutForRegToRegConversion(&ctx, /*tensorShape=*/{64},
                                                  /*repShape=*/{64},
                                                  /*order=*/{0}),
            LinearLayout({{S("offset"), {{1}, {2}, {4}, {8}, {16}, {32}}},
                          {S("iteration"), {}},
                          {S("block"), {}}},
                         {S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, ChooseShmemLayout_Empty) {
  LinearLayout ll = LinearLayout({{S("register"), {{0}}},
                                  {S("lane"), {{0}}},
                                  {S("warp"), {{0}}},
                                  {S("block"), {}}},
                                 {S("dim0")});
  EXPECT_EQ(
      chooseShemLayoutForRegToRegConversion(&ctx, /*tensorShape=*/{},
                                            /*repShape=*/{}, /*order=*/{}),
      LinearLayout({{S("offset"), {}}, {S("iteration"), {}}, {S("block"), {}}},
                   {}));
}

TEST_F(LinearLayoutConversionsTest, ChooseShmemLayout_Multidim) {
  LinearLayout src(
      {{S("register"), {}},
       {S("lane"),
        {{0, 0, 1, 0}, {0, 0, 2, 0}, {1, 0, 0, 0}, {2, 0, 0, 0}, {0, 0, 0, 1}}},
       {S("warp"), {{0, 0, 0, 2}, {0, 1, 0, 0}, {0, 2, 0, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1"), S("dim2"), S("dim3")});
  EXPECT_EQ(
      chooseShemLayoutForRegToRegConversion(&ctx, /*tensorShape=*/{4, 4, 4, 4},
                                            /*repShape=*/{2, 2, 2, 2},
                                            /*order=*/{3, 2, 1, 0}),
      LinearLayout({{S("offset"),
                     {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}},
                    {S("iteration"),
                     {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 2}}},
                    {S("block"), {}}},
                   {S("dim3"), S("dim2"), S("dim1"), S("dim0")}));
}

TEST_F(LinearLayoutConversionsTest, MMAv5Fp4Padded) {
  auto ll = toLinearLayout({32, 64}, nvmmaShared(128, false, 8, {1, 1}, {1, 1},
                                                 {1, 0}, {1, 0}, true));
  EXPECT_EQ(ll, LinearLayout(
                    {{S("offset"),
                      {{0, 1},
                       {0, 2},
                       {0, 4},
                       {0, 0}, // offset 8 maps to the same indices as offset 0
                       {0, 8},
                       {0, 16},
                       {0, 32},
                       {1, 8},
                       {2, 16},
                       {4, 32},
                       {8, 0},
                       {16, 0}}},
                     {S("block"), {}}},
                    {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, TensorMemory_blockM_64) {
  auto enc = tmem(64, 64);
  auto d0 = S("dim0");
  auto d1 = S("dim1");
  auto kRow = S("row");
  auto kCol = S("col");
  auto kBlock = S("block");
  LinearLayout expected1 = LinearLayout(
      {{kRow, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {64, 0}, {16, 0}, {32, 0}}},
       {kCol, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {0, 32}}}},
      {d0, d1});
  expected1 *= LinearLayout::identity1D(1, kBlock, d0);
  EXPECT_EQ(toLinearLayout({128, 64}, enc), expected1);
  // Tensor just fits blockMxblockN -> the layout is not injective (row=16 is
  // zero)
  LinearLayout expected2 = LinearLayout(
      {{kRow, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}, {16, 0}, {32, 0}}},
       {kCol, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {0, 32}}}},
      {d0, d1});
  expected2 *= LinearLayout::identity1D(1, kBlock, d0);
  EXPECT_EQ(toLinearLayout({64, 64}, enc), expected2);
  // Broadcasts M then N
  LinearLayout expected3 = LinearLayout(
      {{kRow, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {64, 0}, {16, 0}, {32, 0}}},
       {kCol,
        {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {0, 32}, {128, 0}, {0, 64}}}},
      {d0, d1});
  expected3 *= LinearLayout::identity1D(1, kBlock, d0);
  EXPECT_EQ(toLinearLayout({256, 128}, enc), expected3);
  // Fits N in basis the 5th basis if shape[0] == 64
  LinearLayout expected4 = LinearLayout(
      {{kRow, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 64}, {16, 0}, {32, 0}}},
       {kCol, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {0, 32}, {0, 128}}}},
      {d0, d1});
  expected4 *= LinearLayout::identity1D(1, kBlock, d0);
  EXPECT_EQ(toLinearLayout({64, 256}, enc), expected4);
}

TEST_F(LinearLayoutConversionsTest, TensorMemory_blockM_128) {
  auto enc = tmem(128, 128);
  auto d0 = S("dim0");
  auto d1 = S("dim1");
  auto kRow = S("row");
  auto kCol = S("col");
  auto kBlock = S("block");
  LinearLayout tile = LinearLayout::identity1D(128, kRow, d0) *
                      LinearLayout::identity1D(128, kCol, d1) *
                      LinearLayout::identity1D(1, kBlock, d0);
  EXPECT_EQ(toLinearLayout({128, 128}, enc), tile);
  EXPECT_EQ(toLinearLayout({256, 128}, enc),
            tile * LinearLayout::identity1D(2, kCol, d0));
  EXPECT_EQ(toLinearLayout({256, 256}, enc),
            tile * LinearLayout::identity1D(2, kCol, d0) *
                LinearLayout::identity1D(2, kCol, d1));
}

TEST_F(LinearLayoutConversionsTest, TensorMemory_fp4Padded) {
  auto enc = TensorMemoryEncodingAttr::get(
      &ctx, 128, 64, 1, CGAEncodingAttr::get1CTALayout(&ctx, 2),
      /*twoCTAs=*/false, /*fp4Padded=*/true);
  auto d0 = S("dim0");
  auto d1 = S("dim1");
  auto kRow = S("row");
  auto kCol = S("col");
  auto kBlock = S("block");
  LinearLayout tile = LinearLayout::identity1D(128, kRow, d0) *
                      LinearLayout::zeros1D(2, kCol, d1) *
                      LinearLayout::identity1D(64, kCol, d1) *
                      LinearLayout::identity1D(1, kBlock, d0);
  EXPECT_EQ(toLinearLayout({128, 64}, enc), tile);
}

TEST_F(LinearLayoutConversionsTest, TensorMemory_CTASplit) {
  auto d0 = S("dim0");
  auto d1 = S("dim1");
  auto kBlock = S("block");
  LinearLayout ll = LinearLayout({{kBlock, {{0, 1}}}}, {d0, d1});
  auto cgaLayout = CGAEncodingAttr::get(&ctx, std::move(ll));
  auto enc = tmem(128, 64, cgaLayout);
  auto enc1 = tmem(128, 64);
  EXPECT_EQ(toLinearLayout({128, 128}, enc),
            toLinearLayout({128, 64}, enc1) *
                LinearLayout::identity1D(2, kBlock, d1));
}

// Tests for SM120 DotScaled Scale Layout
TEST_F(LinearLayoutConversionsTest, SM120DotScaledScaleLayout) {
  LinearLayout layout, ll;

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{128, 2}, /*opIdx=*/0, /*warpsPerCTA=*/{1, 1},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout({{S("register"), {{0, 1}, {16, 0}, {32, 0}, {64, 0}}},
                     {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
                     {S("warp"), {}},
                     {S("block"), {}}},
                    {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{128, 2}, /*opIdx=*/1, /*warpsPerCTA=*/{1, 1},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {8, 0}, {16, 0}, {32, 0}, {64, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{128, 4}, /*opIdx=*/0, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout({{S("register"), {{0, 1}, {0, 2}, {32, 0}, {64, 0}}},
                     {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
                     {S("warp"), {{0, 0}, {16, 0}}},
                     {S("block"), {}}},
                    {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 4}, /*opIdx=*/1, /*warpsPerCTA=*/{1, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {0, 2}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{8, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{128, 8}, /*opIdx=*/0, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll =
      LinearLayout({{S("register"), {{0, 1}, {0, 2}, {0, 4}, {32, 0}, {64, 0}}},
                    {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {{0, 0}, {16, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{128, 8}, /*opIdx=*/1, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {16, 0}, {32, 0}, {64, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{8, 0}, {0, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 2}, /*opIdx=*/0, /*warpsPerCTA=*/{1, 1},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 2}, /*opIdx=*/1, /*warpsPerCTA=*/{1, 1},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {8, 0}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 4}, /*opIdx=*/0, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {0, 2}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{0, 0}, {16, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 4}, /*opIdx=*/1, /*warpsPerCTA=*/{1, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {0, 2}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{8, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 8}, /*opIdx=*/0, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{0, 0}, {16, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);

  layout = getSM120DotScaledScaleLayout(
      &ctx, /*shape=*/{256, 8}, /*opIdx=*/1, /*warpsPerCTA=*/{2, 2},
      /*cgaLayout=*/
      CGAEncodingAttr::fromSplitParams(&ctx, {1, 1}, {1, 1}, {1, 0}));
  ll = LinearLayout(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {16, 0}, {32, 0}, {64, 0}, {128, 0}}},
       {S("lane"), {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {{8, 0}, {0, 0}}},
       {S("block"), {}}},
      {S("dim0"), S("dim1")});

  EXPECT_EQ(ll, layout);
}

//===----------------------------------------------------------------------===//
// nvmmaSharedToLinearLayout TMA Mode Independence Tests
//
// Verify that nvmmaSharedToLinearLayout produces the same result regardless
// of TMA mode. This is critical because MMA lowering uses toLinearLayout()
// to read from shared memory, and it doesn't know which TMA mode was used
// to load the data. If the layouts differ, MMA would compute wrong addresses.
//
// Note: We only test non-transposed encodings because TMA descriptors cannot
// be transposed (see AsyncTMACopyGlobalToLocalOp verification which emits
// "TMA descriptor layout must not be transposed"). Transposed layouts are
// created after TMA load or used for conceptual access patterns, not for
// TMA descriptor configuration.
//===----------------------------------------------------------------------===//

TEST_F(LinearLayoutConversionsTest,
       NvmmaSharedToLinearLayout_TMAModeIndependence) {
  // Test various non-transposed shapes and configurations to ensure the shared
  // memory layout is independent of TMA mode.
  //
  // Test matrix:
  // - swizzleSizeInBytes: 0, 32, 64, 128
  // - non-contiguous dim (dim0): 512, 1024 (exceeds Tiled mode limit of 256)
  // - contiguous dim (dim1): large enough for multiple messages

  constexpr int elementBitWidth = 16; // f16
  constexpr int elementBytes = elementBitWidth / 8;

  for (int swizzleBytes : {0, 32, 64, 128}) {
    for (int64_t dim0 : {512, 1024}) {
      // For contiguous dim, use a size that requires multiple messages.
      // With swizzle, the contiguous dim block size = swizzleBytes / elemBytes.
      // Use 2x the max swizzle size to ensure multiple messages in dim1.
      int64_t dim1 = (swizzleBytes == 0) ? 64 : (128 / elementBytes) * 2;

      auto encoding =
          nvmmaShared(swizzleBytes, /*transposed=*/false, elementBitWidth,
                      {1, 1}, {1, 1}, {1, 0}, {1, 0});
      llvm::SmallVector<int64_t> shape = {dim0, dim1};

      auto tiledLayout =
          nvmmaSharedToLinearLayout(shape, encoding, TMAMode::Tiled);
      auto im2colLayout =
          nvmmaSharedToLinearLayout(shape, encoding, TMAMode::Im2Col);

      EXPECT_EQ(tiledLayout, im2colLayout)
          << "Shared memory layout must be independent of TMA mode for shape ["
          << dim0 << ", " << dim1 << "] with " << swizzleBytes << "B swizzle";
    }
  }
}

} // anonymous namespace
} // namespace mlir::triton::gpu

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
