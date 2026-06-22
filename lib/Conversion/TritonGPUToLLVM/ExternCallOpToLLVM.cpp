#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#ifdef alloca
#undef alloca
#endif

using namespace mlir;
using namespace mlir::triton;

namespace {

SmallVector<SmallVector<int32_t>>
extractBases(const LinearLayout &ll, StringAttr inDim) {
  auto bases = ll.getBases();
  auto it = bases.find(inDim);
  if (it == bases.end())
    return {};
  SmallVector<SmallVector<int32_t>> result;
  for (auto &row : it->second)
    result.push_back(SmallVector<int32_t>(row.begin(), row.end()));
  return result;
}

std::string
computeLayoutHash(const std::string &symbol,
                  ArrayRef<triton::gpu::ExternCallOp> ops,
                  MLIRContext *ctx) {
  std::string key = symbol;
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane     = StringAttr::get(ctx, "lane");
  StringAttr kWarp     = StringAttr::get(ctx, "warp");

  for (auto op : ops) {
    for (auto operand : op.getInputs()) {
      auto tensorTy = cast<RankedTensorType>(operand.getType());
      auto encoding = tensorTy.getEncoding();
      auto shape = tensorTy.getShape();
      LinearLayout ll = triton::gpu::toLinearLayout(shape, encoding);
      key += "|";
      key += std::to_string(ll.getInDimSize(kRegister));
      key += ",";
      key += std::to_string(ll.getInDimSize(kLane));
      key += ",";
      key += std::to_string(ll.getInDimSize(kWarp));
    }
  }
  return llvm::formatv("{0:x}", llvm::hash_value(key)).str();
}

void
storeLayoutMetadata(ModuleOp module,
                    const std::string &mangledName,
                    const std::string &libpath,
                    const std::string &symbol,
                    ArrayRef<triton::gpu::ExternCallOp> ops) {
  auto *ctx = module.getContext();
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane     = StringAttr::get(ctx, "lane");
  StringAttr kWarp     = StringAttr::get(ctx, "warp");

  std::string jsonStr;
  llvm::raw_string_ostream os(jsonStr);

  // Read existing JSON if any, or start a new dict
  auto existing = module->getAttrOfType<StringAttr>("ttg.extern_call_specs");
  if (existing) {
    StringRef old = existing.getValue();
    // Strip the trailing '}'
    os << old.substr(0, old.size() - 1) << ",\n";
  } else {
    os << "{\n";
  }

  // Append new entry
  os << "  \"" << mangledName << "\": {\n";
  os << "    \"spec\": {\"libpath\": \"" << libpath << "\", \"symbol\": \"" << symbol << "\"},\n";
  os << "    \"inputs\": [\n";

  bool firstInput = true;
  for (auto op : ops) {
    for (auto operand : op.getInputs()) {
      if (!firstInput) os << ",\n";
      firstInput = false;

      auto tensorTy = cast<RankedTensorType>(operand.getType());
      auto encoding = tensorTy.getEncoding();
      auto shape = tensorTy.getShape();
      LinearLayout ll = triton::gpu::toLinearLayout(shape, encoding);

      auto regBases = extractBases(ll, kRegister);
      auto laneBases = extractBases(ll, kLane);
      auto warpBases = extractBases(ll, kWarp);

      auto flatten = [](const SmallVector<SmallVector<int32_t>> &bases) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << "[";
        bool first = true;
        for (auto &row : bases) {
          for (auto v : row) {
            if (!first) ss << ", ";
            first = false;
            ss << v;
          }
        }
        ss << "]";
        return s;
      };

      os << "      {\n";
      os << "        \"shape\": [";
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) os << ", ";
        os << shape[i];
      }
      os << "],\n";
      os << "        \"num_warps\": " << ll.getInDimSize(kWarp) << ",\n";
      os << "        \"reg_bases\": " << flatten(regBases) << ",\n";
      os << "        \"lane_bases\": " << flatten(laneBases) << ",\n";
      os << "        \"warp_bases\": " << flatten(warpBases) << ",\n";

      auto elemTy = tensorTy.getElementType();
      const char *dtype = "f32";
      if (elemTy.isF32()) dtype = "f32";
      else if (elemTy.isF64()) dtype = "f64";
      else if (elemTy.isF16()) dtype = "f16";
      else if (elemTy.isBF16()) dtype = "bf16";
      else if (elemTy.isInteger(32)) dtype = "i32";
      else if (elemTy.isInteger(64)) dtype = "i64";
      else if (elemTy.isInteger(8)) dtype = "i8";
      os << "        \"dtype\": \"" << dtype << "\"\n";
      os << "      }";
    }
  }

  os << "\n    ]\n";
  os << "  }\n";
  os << "}";

  module->setAttr("ttg.extern_call_specs",
                  StringAttr::get(ctx, os.str()));
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

    std::string mangledName = "__triton_ext_" + op.getSymbol().str() + "_" +
                               computeLayoutHash(op.getSymbol().str(), {op}, ctx);

    storeLayoutMetadata(module, mangledName, op.getLibpath().str(),
                        op.getSymbol().str(), {op});

    auto caller = op->template getParentOfType<FunctionOpInterface>();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto promotedOperands = this->getTypeConverter()->promoteOperands(
        loc, /*opOperands=*/op->getOperands(), adaptor.getOperands(),
        rewriter);

    // Convert tensor struct operands to pointers: alloca + store.
    // NVPTX passes large structs by value, but the CUDA wrapper
    // compiled by clang accepts them as pointers. We allocate
    // space on the stack to create a pointer for each struct.
    unsigned numTensorArgs = op.getInputs().size();
    for (unsigned i = 0; i < numTensorArgs; ++i) {
      auto structVal = promotedOperands[i];
      auto structTy = structVal.getType();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx, 0);
      Value one = b.i32_val(1);
      // Use LLVM::AllocaOp::create to create a stack allocation
      // (avoiding name conflict with C library alloca() macro)
      auto *builder = &static_cast<OpBuilder &>(rewriter);
      Value stackPtr = LLVM::AllocaOp::create(
          *builder, loc, ptrTy, structTy, one, 0).getResult();
      b.store(structVal, stackPtr);
      promotedOperands[i] = stackPtr;
    }
    if (!caller->hasAttr("allocation.offset") ||
        !op->hasAttr("allocation.offset")) {
      auto base = LLVM::getStackPointer(rewriter, caller);
      auto i32Type = rewriter.getIntegerType(32);
      promotedOperands.push_back(b.ptrtoint(i32Type, base));
    } else {
      auto base =
          LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
      auto i32Type = rewriter.getIntegerType(32);
      promotedOperands.push_back(b.ptrtoint(i32Type, base));
    }

    auto opOffsetAttr = op->getAttrOfType<mlir::IntegerAttr>(
        "ttg.global_scratch_memory_offset");
    Value globalOffsetVal;
    if (opOffsetAttr)
      globalOffsetVal = b.i32_val(opOffsetAttr.getValue().getZExtValue());
    Value gscratch = LLVM::getGlobalScratchPtr(
        loc, rewriter, targetInfo, caller, globalOffsetVal);
    promotedOperands.push_back(
        b.addrspacecast(LLVM::LLVMPointerType::get(ctx, 0), gscratch));

    auto profileOffsetAttr = op->getAttrOfType<mlir::IntegerAttr>(
        "ttg.profile_scratch_memory_offset");
    Value profileOffsetVal;
    if (profileOffsetAttr)
      profileOffsetVal = b.i32_val(profileOffsetAttr.getValue().getZExtValue());
    Value pscratch = LLVM::getProfileScratchPtr(
        loc, rewriter, targetInfo, caller, profileOffsetVal);
    promotedOperands.push_back(
        b.addrspacecast(LLVM::LLVMPointerType::get(ctx, 0), pscratch));

    // Create an output buffer (alloca) and prepend its pointer
    // to the argument list. The wrapper writes the result struct
    // into this buffer, and we load from it after the call.
    Value outPtr;
    unsigned numResults = op.getNumResults();
    auto resultTypes = llvm::to_vector<4>(op.getResultTypes());
    Type packedResult = nullptr;
    if (numResults != 0) {
      packedResult =
          this->getTypeConverter()->packFunctionResults(resultTypes);
      if (!packedResult)
        return failure();
      auto ptrTy = LLVM::LLVMPointerType::get(ctx, 0);
      Value one = b.i32_val(1);
      auto *builder = &static_cast<OpBuilder &>(rewriter);
      outPtr = LLVM::AllocaOp::create(
          *builder, loc, ptrTy, packedResult, one, 0).getResult();
      promotedOperands.insert(promotedOperands.begin(), outPtr);
    }

    // Function return type is always void — the wrapper writes
    // the result into the output buffer passed as first argument.
    SmallVector<Type> promotedTypes;
    for (auto v : promotedOperands)
      promotedTypes.push_back(v.getType());
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), promotedTypes);

    auto funcOp = appendOrGetExternFuncOp(rewriter, op, mangledName, funcType,
                                          "", op.getLibpath());

    auto callOp = LLVM::CallOp::create(rewriter, loc, funcOp, promotedOperands);
    callOp.getProperties().setOpBundleSizes(
        rewriter.getDenseI32ArrayAttr({}));
    callOp.getProperties().setOperandSegmentSizes(
        {static_cast<int>(promotedOperands.size()), 0});

    SmallVector<Value> results;
    if (numResults != 0) {
      auto structResult = b.load(packedResult, outPtr);
      if (numResults < 2) {
        results.push_back(structResult);
      } else {
        for (unsigned i = 0; i < numResults; ++i) {
          results.push_back(LLVM::ExtractValueOp::create(
              rewriter, loc, structResult, i));
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
