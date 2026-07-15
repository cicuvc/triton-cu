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

static LogicalResult getExtractorNames(ModuleOp module,
                                        const std::string &symbol,
                                std::vector<std::string> &names) {
  auto attr = module->getAttrOfType<StringAttr>(
      "ttg.extern_call_extractor_names");
  if (!attr)
    return success();

  auto json = llvm::json::parse(attr.getValue());
  if (!json)
    return success();

  auto *obj = json->getAsObject();
  if (!obj)
    return success();

  auto it = obj->find(symbol);
  if (it == obj->end())
    return success();

  auto arr = it->second.getAsArray();
  if (!arr)
    return success();

  for (auto &v : *arr)
    if (auto s = v.getAsString())
      names.push_back(s->str());
  return success();
}

static LogicalResult getArgMemorySpaces(ModuleOp module,
    const std::string &symbol, std::vector<std::string> &spaces) {
  auto attr = module->getAttrOfType<StringAttr>(
      "ttg.extern_call_arg_spaces");
  if (!attr)
    return success();

  auto json = llvm::json::parse(attr.getValue());
  if (!json)
    return success();

  auto *obj = json->getAsObject();
  if (!obj)
    return success();

  auto it = obj->find(symbol);
  if (it == obj->end())
    return success();

  auto arr = it->second.getAsArray();
  if (!arr)
    return success();

  for (auto &v : *arr) {
    auto s = v.getAsString();
    spaces.push_back(s ? s->str() : "register");
  }
  return success();
}

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

static Type buildClangPackedReturnType(MLIRContext *ctx, Type packedResult) {
  auto packedST = cast<LLVM::LLVMStructType>(packedResult);
  SmallVector<Type> clangFields;
  for (auto field : packedST.getBody())
    clangFields.push_back(buildClangStructType(ctx, field));
  return LLVM::LLVMStructType::getLiteral(ctx, clangFields);
}

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
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 0);

    // D-16: Read per-arg memory spaces from module attribute.
    std::vector<std::string> argSpaces;
    (void)getArgMemorySpaces(module, op.getSymbol().str(), argSpaces);
    if (argSpaces.empty()) {
      // Fallback for pre-Phase-6 modules: all operands are register.
      argSpaces.assign(numTensorArgs, "register");
    }

    for (unsigned i = 0; i < numTensorArgs; ++i) {
      bool isShared = (i < argSpaces.size() && argSpaces[i] == "shared");
      if (isShared) {
        // SHLOWER-01/02: bypass alloca+store+ptr path.
        // Extract shared memory object from promoted memdesc struct,
        // apply subview offset via getShmemAffineBase, pass AS3 ptr
        // directly to callee (no stack slot — avoids L-01 AS3 erasure).
        auto memDescType = cast<triton::gpu::MemDescType>(op.getInputs()[i].getType());
        auto llvmElemTy = getTypeConverter()->convertType(
            memDescType.getElementType());
        auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
            loc, promotedOperands[i], llvmElemTy, rewriter);
        promotedOperands[i] = smemObj.getShmemAffineBase(
            loc, rewriter, memDescType);
        // Result is ptr addrspace(3) — passes directly to callee.
      } else {
        // Existing distributed path: alloca + store + ptr (AS0).
        auto structVal = promotedOperands[i];
        auto structTy = structVal.getType();
        Value one = b.i32_val(1);
        auto *builder = &static_cast<OpBuilder &>(rewriter);
        Value stackPtr = LLVM::AllocaOp::create(
            *builder, loc, ptrTy, structTy, one, 0).getResult();
        b.store(structVal, stackPtr);
        promotedOperands[i] = stackPtr;
      }
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

    std::vector<std::string> extractorNames;
    (void)getExtractorNames(module, op.getSymbol().str(), extractorNames);

    if (!extractorNames.empty()) {
      // Tuple return: emit sret call + extractor calls.
      if (extractorNames.size() != numResults)
        return op.emitError("extractor count mismatch for '")
               << op.getSymbol() << "': " << extractorNames.size()
               << " extractors for " << numResults << " results";

      auto packedST = cast<LLVM::LLVMStructType>(packedResult);
      Type clangPackedTy = buildClangPackedReturnType(ctx, packedResult);

      // Allocate sret buffer matching the packed clang struct type.
      Value one = b.i32_val(1);
      auto *builder = &static_cast<OpBuilder &>(rewriter);
      auto mainRET = LLVM::AllocaOp::create(
          *builder, loc, ptrTy, clangPackedTy, one, 0);

      // Build main call: void @fn(ptr sret, ptr arg0, ptr arg1, ...)
      SmallVector<Type> mainArgTypes;
      mainArgTypes.push_back(ptrTy);
      mainArgTypes.append(promotedTypes.begin(), promotedTypes.end());
      auto mainFuncType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(ctx), mainArgTypes);

      auto mainFuncOp = appendOrGetExternFuncOp(
          rewriter, op, mangledName, mainFuncType, "", op.getLibpath());

      SmallVector<Value> mainArgs;
      mainArgs.push_back(mainRET.getResult());
      mainArgs.append(promotedOperands.begin(), promotedOperands.end());

      auto mainCall = LLVM::CallOp::create(rewriter, loc, mainFuncOp, mainArgs);
      mainCall.getProperties().setOpBundleSizes(
          rewriter.getDenseI32ArrayAttr({}));
      mainCall.getProperties().setOperandSegmentSizes(
          {static_cast<int>(mainArgs.size()), 0});

      // For each result, emit extractor call.
      auto nullTag = b.int_val(64, 0);
      SmallVector<Value> results;
      for (unsigned i = 0; i < numResults; ++i) {
        auto flatFieldTy = packedST.getBody()[i];
        auto clangFieldTy = buildClangStructType(ctx, flatFieldTy);
        auto extrFuncType = LLVM::LLVMFunctionType::get(
            clangFieldTy, {ptrTy, ptrTy});

        auto extrFuncOp = appendOrGetExternFuncOp(
            rewriter, op, extractorNames[i], extrFuncType, "", op.getLibpath());

        Value nullTagPtr = b.inttoptr(ptrTy, nullTag);
        SmallVector<Value> extrArgs = {nullTagPtr, mainRET.getResult()};
        auto extrCall = LLVM::CallOp::create(rewriter, loc, extrFuncOp, extrArgs);
        extrCall.getProperties().setOpBundleSizes(
            rewriter.getDenseI32ArrayAttr({}));
        extrCall.getProperties().setOperandSegmentSizes(
            {static_cast<int>(extrArgs.size()), 0});

        auto flat = convertClangToFlat(
            loc, rewriter, extrCall.getResult(), flatFieldTy);
        results.push_back(flat);
      }

      rewriter.replaceOp(op, results);
      return success();
    }

    // Non-tuple path (existing logic).
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
        auto flat = convertClangToFlat(loc, rewriter, clangResult,
                                       packedResult);
        results.push_back(flat);
      } else {
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
