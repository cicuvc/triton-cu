#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

LogicalResult
getMangledName(ModuleOp module, const std::string &symbol,
               std::string &mangledName) {
  auto attr = module->getAttrOfType<StringAttr>(
      "ttg.extern_call_mangled_names");
  if (!attr)
    return failure();

  auto json = llvm::json::parse(attr.getValue());
  if (!json)
    return failure();

  auto *obj = json->getAsObject();
  if (!obj)
    return failure();

  auto it = obj->find(symbol);
  if (it == obj->end())
    return failure();

  auto val = it->second.getAsString();
  if (!val)
    return failure();

  mangledName = val->str();
  return success();
}

// Build a clang-compatible struct type: { [N x elemTy] }
// Clang compiles C++ "T data[REG_SIZE]" into a struct containing an array.
// MLIR's type converter produces flat { scalar x N }. This function
// converts the flat type to the array-containing form.
static LLVM::LLVMStructType
buildClangStructType(MLIRContext *ctx, Type flatStructType) {
  auto flatST = cast<LLVM::LLVMStructType>(flatStructType);
  auto body = flatST.getBody();
  if (body.empty())
    return flatST;
  Type elemTy = body[0];
  unsigned N = body.size();
  auto arrayTy = LLVM::LLVMArrayType::get(elemTy, N);
  return LLVM::LLVMStructType::getLiteral(ctx, {arrayTy});
}

// Convert the packed result type to clang-compatible form.
// Single result:  { elemTy x N }          ->   { [N x elemTy] }
// Multiple:       { {e0 x N0}, {e1 x N1} } ->  { {[N0 x e0]}, {[N1 x e1]} }
static Type buildClangPackedReturnType(MLIRContext *ctx, Type packedResult) {
  auto packedST = cast<LLVM::LLVMStructType>(packedResult);
  SmallVector<Type> clangFields;
  for (auto field : packedST.getBody())
    clangFields.push_back(buildClangStructType(ctx, field));
  return LLVM::LLVMStructType::getLiteral(ctx, clangFields);
}

// After the call, convert a clang struct { [N x elemTy] }
// back to MLIR's flat struct { elemTy x N }.
static Value convertClangToFlat(Location loc,
                                ConversionPatternRewriter &rewriter,
                                Value clangResult, Type flatType) {
  auto flatST = cast<LLVM::LLVMStructType>(flatType);
  auto body = flatST.getBody();
  unsigned N = body.size();

  auto arrVal =
      LLVM::ExtractValueOp::create(rewriter, loc, clangResult, {0});

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value flat = b.undef(flatType);
  for (unsigned i = 0; i < N; ++i) {
    auto elem =
        LLVM::ExtractValueOp::create(rewriter, loc, arrVal, {i});
    flat = LLVM::InsertValueOp::create(rewriter, loc, flat, elem, {i});
  }
  return flat;
}

struct ExternCallOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ExternCallOp> {
  ExternCallOpConversion(LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::ExternCallOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ExternCallOp op,
                  typename triton::gpu::ExternCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();

    auto module = op->template getParentOfType<ModuleOp>();

    std::string mangledName;
    if (failed(getMangledName(module, op.getSymbol().str(), mangledName))) {
      return op.emitError("extern_call symbol '")
             << op.getSymbol() << "' not found in pre-compiled mangled names";
    }

    auto caller = op->template getParentOfType<FunctionOpInterface>();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto promotedOperands = this->getTypeConverter()->promoteOperands(
        loc, /*opOperands=*/op->getOperands(), adaptor.getOperands(),
        rewriter);

    unsigned numTensorArgs = op.getInputs().size();
    for (unsigned i = 0; i < numTensorArgs; ++i) {
      auto structVal = promotedOperands[i];
      auto structTy = structVal.getType();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx, 0);
      Value one = b.i32_val(1);
      auto *builder = &static_cast<OpBuilder &>(rewriter);
      Value stackPtr = LLVM::AllocaOp::create(
          *builder, loc, ptrTy, structTy, one, 0).getResult();
      b.store(structVal, stackPtr);
      promotedOperands[i] = stackPtr;
    }

    SmallVector<Type> promotedTypes;
    for (auto v : promotedOperands)
      promotedTypes.push_back(v.getType());

    unsigned numResults = op.getNumResults();
    Type packedResult = nullptr;
    if (numResults != 0) {
      auto resultTypes = llvm::to_vector<4>(op.getResultTypes());
      packedResult =
          this->getTypeConverter()->packFunctionResults(resultTypes);
      if (!packedResult)
        return failure();
    }

    // Build clang-compatible return type: { [N x elemTy] } instead
    // of MLIR's flat { elemTy x N }.
    // For single result, packedResult IS the flat tensor struct.
    // For multiple results, packedResult wraps flattened tensor structs.
    Type clangReturnType;
    if (packedResult) {
      if (numResults < 2) {
        clangReturnType = buildClangStructType(ctx, packedResult);
      } else {
        clangReturnType = buildClangPackedReturnType(ctx, packedResult);
      }
    } else {
      clangReturnType = LLVM::LLVMVoidType::get(ctx);
    }

    auto funcType =
        LLVM::LLVMFunctionType::get(clangReturnType, promotedTypes);

    auto funcOp = appendOrGetExternFuncOp(rewriter, op, mangledName, funcType,
                                          "", op.getLibpath());

    auto callOp = LLVM::CallOp::create(rewriter, loc, funcOp, promotedOperands);
    callOp.getProperties().setOpBundleSizes(
        rewriter.getDenseI32ArrayAttr({}));
    callOp.getProperties().setOperandSegmentSizes(
        {static_cast<int>(promotedOperands.size()), 0});

    SmallVector<Value> results;
    if (numResults != 0) {
      Value clangResult = callOp.getResult();
      if (numResults < 2) {
        // Single tensor result: convert { [N x elemTy] } → { elemTy x N }
        auto flat = convertClangToFlat(loc, rewriter, clangResult,
                                       packedResult);
        results.push_back(flat);
      } else {
        // Multiple tensor results: packedResult is
        // { flat0, flat1, ... } and clangResult is
        // { clang0, clang1, ... }. Extract each clang field
        // and convert to flat.
        auto packedST = cast<LLVM::LLVMStructType>(packedResult);
        for (unsigned i = 0; i < packedST.getBody().size(); ++i) {
          auto clangField = LLVM::ExtractValueOp::create(
              rewriter, loc, clangResult, {i});
          auto flat = convertClangToFlat(loc, rewriter, clangField,
                                         packedST.getBody()[i]);
          results.push_back(flat);
        }
      }
    }

    rewriter.replaceOp(op, results);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateExternCallOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ExternCallOpConversion>(typeConverter, targetInfo, benefit);
}
