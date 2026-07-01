#pragma once

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// Forward-declare MLIR type for the extraction API.
namespace mlir {
class ModuleOp;
}

enum class ScalarType { Int32, Int64, Fp32, Fp16, Bf16, Fp8e4m3, Fp8e5m2 };

struct TensorParameter {
    ScalarType Type;
    std::vector<uint32_t> Shape;
    std::vector<uint32_t> LayoutShape;
    std::vector<uint32_t> RegBasis;
    std::vector<uint32_t> LaneBasis;
    std::vector<uint32_t> WarpBasis;
    uint32_t N_WARPS;
};

struct DeviceFunctionInstantiation {
    std::string FunctionLookupName;
    std::vector<std::variant<ScalarType, TensorParameter>> ParamTypes;
    std::optional<llvm::Function *> InstFunction;
};

struct CudaInstantiatedFunc {
    std::string Symbol;
    std::string MangledName;
};

// Public API — defined in clang_compiler.cc
// Returns (bitcode, error, results). bitcode is the compiled LLVM module
// serialized; use linkBitcodeToModule to merge into another module.
std::tuple<std::string, std::string, std::vector<CudaInstantiatedFunc>>
tritonCompileCuda(llvm::LLVMContext &ctx, const std::string &source,
                  const std::string &sm, const std::string &resourceDir,
                  const std::vector<std::string> &includePaths,
                  std::vector<DeviceFunctionInstantiation> &instantiations);

std::string tritonExtractExternCallSpecs(mlir::ModuleOp module);

// Links compiled CUDA bitcode into dstMod (same LLVMContext).
void linkBitcodeToModule(llvm::Module *dstMod, const std::string &bitcode,
                         llvm::LLVMContext &ctx);
