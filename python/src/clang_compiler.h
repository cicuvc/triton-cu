#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/GlobalDecl.h>
#include <clang/AST/TemplateBase.h>
#include <clang/AST/TemplateName.h>
#include <clang/AST/TypeBase.h>
#include <clang/Basic/IdentifierTable.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Stack.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Sema/Sema.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/VirtualFileSystem.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Forward-declare MLIR type for the extraction API.
namespace mlir {
class ModuleOp;
class Type;
}

// ============================================================
// ABI abstraction for cross-architecture context switching
// ============================================================

struct ExecutionContext;

using SwitchContextFn = void (*)(uint64_t *saveGPR,
                                 const uint64_t *restoreGPR);

struct TargetABI {
  const char *Name;
  unsigned GPRCount;
  unsigned StackAlignment;
  struct {
    unsigned IP, SP, BP, Arg0, Self;
  } Slots;
  SwitchContextFn SwitchContext;
};

inline constexpr unsigned kMaxGPR = 32;

extern const TargetABI X64SysVABI;

struct ExecutionContext {
  static constexpr unsigned MaxGPR = kMaxGPR;
  const TargetABI *ABI;
  uint64_t GPR[MaxGPR]{};

  void (*Entry)(uint64_t, ExecutionContext &);
  ExecutionContext *ExitWay;

  char *StackStart;
  char *StackBase;

  ExecutionContext()
      : ABI{nullptr}, StackStart{nullptr}, StackBase{nullptr} {}
  ExecutionContext(const TargetABI &abi)
      : ABI{&abi}, StackStart{nullptr}, StackBase{nullptr} {}
  ExecutionContext(const TargetABI &abi,
                   void (*Entry)(uint64_t, ExecutionContext &),
                   ExecutionContext *ExitWay, uint64_t Arg0,
                   uint64_t StackSize)
      : ABI{&abi}, Entry{Entry}, ExitWay{ExitWay} {
    StackSize = ((StackSize - 1) / abi.StackAlignment + 1) *
                abi.StackAlignment;
    StackStart = reinterpret_cast<char *>(
        std::aligned_alloc(abi.StackAlignment, StackSize));
    StackBase = StackStart + StackSize - 8;

    GPR[abi.Slots.IP] = (uint64_t)(LaunchPad);
    GPR[abi.Slots.Arg0] = Arg0;
    GPR[abi.Slots.Self] = (uint64_t)this;
    GPR[abi.Slots.BP] = (uint64_t)StackBase;
    GPR[abi.Slots.SP] = (uint64_t)StackBase - 8;
  }
  ExecutionContext(const ExecutionContext &) = delete;
  ExecutionContext(ExecutionContext &&) noexcept = delete;
  ExecutionContext &operator=(const ExecutionContext &) = delete;
  ExecutionContext &operator=(ExecutionContext &&) noexcept = delete;
  ~ExecutionContext() {
    if (StackStart)
      std::free(StackStart);
  }
  static void LaunchPad(uint64_t Arg0, ExecutionContext &Self) {
    Self.Entry(Arg0, Self);
    Self.SwitchTo(*Self.ExitWay);
  }
  __attribute__((noinline)) void
  SwitchTo(const ExecutionContext &DstCtx) {
    ABI->SwitchContext(GPR, DstCtx.GPR);
  }
};

// ============================================================
// Domain data types
// ============================================================

struct Dims {
  unsigned RANK = 0;
  uint64_t SIZE = 1;
};

enum class ScalarType { Int32, Int64, Fp32, Fp16, Bf16, Fp8e4m3, Fp8e5m2 };

struct LayoutInfo {
  std::vector<uint32_t> LayoutShape;
  std::vector<uint32_t> RegBasis;
  std::vector<uint32_t> LaneBasis;
  std::vector<uint32_t> WarpBasis;
  uint32_t N_WARPS = 0;
};

struct TensorParameter {
  ScalarType Type;
  std::vector<uint32_t> Shape;
  LayoutInfo Layout;
};

struct TupleType {
  std::vector<std::variant<std::nullptr_t, TensorParameter, TupleType>>
      Types;
};

struct ShapeResult {
  clang::ClassTemplateSpecializationDecl *spec = nullptr;
  clang::QualType type;
  Dims dims;
};

struct LayoutFactoryContext {
  clang::ClassTemplateSpecializationDecl *spec = nullptr;
  clang::ClassTemplateDecl *BasisGroupTmpl = nullptr;
  clang::ClassTemplateDecl *LayoutTmpl = nullptr;
  unsigned N_WARPS = 0, N_LANE_AXES = 0, N_REG_AXES = 0, N_WARP_AXES = 0;
};

// ============================================================
// API types (Python-bound)
// ============================================================

struct CudaFuncRequest {
  std::string Symbol;
  std::vector<std::variant<ScalarType, TensorParameter>> ParamTypes;
  bool UseFastMath = false;
};

struct CudaFuncResult {
  std::string Symbol;
  std::string MangledName;
  std::vector<TensorParameter> ReturnTypes;
  std::vector<std::string> ExtractorMangledNames;
};

// ============================================================
// Utility free functions
// ============================================================

inline ScalarType getScalarTypeFromQualType(clang::ASTContext &Ctx,
                                             clang::QualType Type) {
  if (Type.getCanonicalType() == Ctx.FloatTy)
    return ScalarType::Fp32;
  if (Type.getCanonicalType() == Ctx.HalfTy)
    return ScalarType::Fp16;
  if (Type.getCanonicalType() == Ctx.IntTy)
    return ScalarType::Int32;
  if (Type.getCanonicalType() == Ctx.LongLongTy)
    return ScalarType::Int64;
  __builtin_unreachable();
}

inline clang::ClassTemplateDecl *
getTemplateDecl(clang::ASTContext &Ctx, const char *Name) {
  auto R = Ctx.getTranslationUnitDecl()->lookup(&Ctx.Idents.get(Name));
  if (R.empty())
    return nullptr;
  return llvm::dyn_cast<clang::ClassTemplateDecl>(*R.begin());
}

inline clang::QualType getQualTypeFromScalarType(clang::ASTContext &Ctx,
                                                  ScalarType type) {
  switch (type) {
  case ScalarType::Fp32:
    return Ctx.FloatTy;
  case ScalarType::Fp16:
    return Ctx.HalfTy;
  case ScalarType::Int32:
    return Ctx.IntTy;
  case ScalarType::Int64:
    return Ctx.LongLongTy;
  default:
    assert(false && "unsupported scalar type");
  }
  __builtin_unreachable();
}

// ============================================================
// TypeBuilder — forward: user data → Clang AST types
// ============================================================

struct TypeBuilder {
  clang::ASTContext &Ctx;
  clang::Sema &SemaRef;
  clang::ClassTemplateDecl *ShapeTemplateType;
  clang::ClassTemplateDecl *LayoutFactoryTemplateType;
  clang::ClassTemplateDecl *IntTupleTemplateType;
  clang::ClassTemplateDecl *IntsTemplateType;
  clang::ClassTemplateDecl *TensorTemplateType;

  TypeBuilder(clang::ASTContext &Ctx, clang::Sema &S);
  clang::TemplateArgument mkIntegralArgUint32(uint32_t V);
  clang::TemplateArgument mkTypeArg(clang::QualType T);
  std::optional<uint32_t>
  EvaluateConstexpr(clang::ClassTemplateSpecializationDecl *Spec,
                    const llvm::StringRef &Name);
  ShapeResult buildShape(llvm::ArrayRef<uint32_t> shapeDims);
  clang::ClassTemplateSpecializationDecl *
  BuildIntTuple(clang::SourceLocation SL, unsigned N);
  clang::QualType BuildInts(uint32_t N);
  LayoutFactoryContext BuildLayoutFactory(const ShapeResult &shape,
                                          uint32_t N_WARPS);
  std::pair<std::optional<clang::TemplateArgument>,
            clang::ClassTemplateSpecializationDecl *>
  BuildBasisGroup(const LayoutFactoryContext &LF, unsigned N_BASES,
                  llvm::SmallVector<uint32_t, 4> vecs);
  clang::QualType BuildLayout(const LayoutFactoryContext &LF,
                              clang::TemplateArgument aRegs,
                              clang::TemplateArgument aLanes,
                              clang::TemplateArgument aWarps);
  clang::QualType BuildTensor(clang::QualType ElementType,
                              clang::QualType ShapeType,
                              clang::QualType LayoutType);
};

// ============================================================
// TypeInspector — reverse: Clang AST types → user data
// ============================================================

struct TypeInspector {
  clang::ASTContext &Ctx;
  clang::ClassTemplateDecl *TensorTemplateType;

  TypeInspector(clang::ASTContext &Ctx);
  uint32_t
  EvaulateConstantTemplateNTTP(const clang::TemplateArgument &Arg);
  llvm::SmallVector<uint32_t, 4> ParseShapeType(clang::QualType type);
  llvm::SmallVector<uint32_t, 4>
  ParseBasis(const clang::TemplateArgument &Arg);
  LayoutInfo ParseLayoutType(clang::QualType type);
  TensorParameter
  ParseTensorType(clang::ClassTemplateSpecializationDecl *type);
  TupleType
  ParseTupleType(clang::ClassTemplateSpecializationDecl *type);
  std::variant<std::nullptr_t, TensorParameter, TupleType>
  DispatchTypeParsing(clang::QualType type);
};

// ============================================================
// FunctionResolver — overload resolution + template instantiation
// ============================================================

struct FunctionResolver {
  clang::ASTContext &Ctx;
  clang::Sema &SemaRef;
  clang::SourceLocation SL;

  FunctionResolver(clang::ASTContext &Ctx, clang::Sema &S);
  clang::FunctionDecl *
  LookupFunction(const llvm::StringRef &Name,
                 const llvm::ArrayRef<clang::QualType> &ArgumentTypes);
  clang::FunctionDecl *InstantiateFunction(clang::FunctionDecl *FD);
};

// ============================================================
// TensorTypeHelpers — thin facade
// ============================================================

struct TensorTypeHelpers {
  TypeBuilder Builder;
  TypeInspector Inspector;
  FunctionResolver Resolver;

  TensorTypeHelpers(clang::ASTContext &Ctx, clang::Sema &S)
      : Builder(Ctx, S), Inspector(Ctx), Resolver(Ctx, S) {}
};

// ============================================================
// CUDACompiler — top-level orchestrator (coroutine-based)
// ============================================================

struct CustomAstConsumer;

struct CUDACompiler {
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> Vfs;
  std::unique_ptr<llvm::WritableMemoryBuffer> SourceBuffer;
  std::unique_ptr<clang::CompilerInstance> CI;

  std::unique_ptr<ExecutionContext> CompileExecutionContext;
  std::unique_ptr<ExecutionContext> InvocationContext;

  std::queue<std::function<void(TensorTypeHelpers &,
                                CustomAstConsumer &)>>
      TaskQueue;

  static void PerformCompileImpl(uint64_t Arg0,
                                 ExecutionContext &ExecCtx);
  CUDACompiler(llvm::StringRef SourceCode, int OptLevel,
               const std::string &sm, const std::string &resourceDir,
               const std::vector<std::string> &includePaths);
  void PerformParse(llvm::LLVMContext &Context,
                    const llvm::StringRef &ModuleName);

  clang::QualType BuildTensor(const TensorParameter &);
  clang::QualType BuildInts(uint32_t N);
  clang::FunctionDecl *
  LookupFunction(const llvm::StringRef &Name,
                 const llvm::ArrayRef<clang::QualType> &Args);
  std::variant<std::nullptr_t, TensorParameter, TupleType>
  EvaluateFunctionReturnType(clang::FunctionDecl *FD);
  llvm::Function *InstantiationFunction(clang::FunctionDecl *);
  std::unique_ptr<llvm::Module> EmitFinalModule();

  // INFER-07: Split-path compile — runs inference+codegen+emit phases
  // on an already-parsed compiler (Parse must have been called first).
  std::tuple<std::string, std::string, std::vector<CudaFuncResult>>
  compileBitcode(const std::vector<CudaFuncRequest> &requests);

  // D-01: Inference-only — runs type inference (Phase 1 of compileBitcode)
  // on an already-parsed compiler without emitting LLVM bitcode.
  // Returns per-request CudaFuncResult with ReturnTypes populated;
  // MangledName/ExtractorMangledNames are empty (no codegen).
  std::tuple<std::string, std::string, std::vector<CudaFuncResult>>
  inferReturnTypes(const std::vector<CudaFuncRequest> &requests);
};

// ============================================================
// Clang ASTConsumer / FrontendAction adapters
// ============================================================

struct CustomAstConsumer : clang::CodeGenerator {
  llvm::LLVMContext &llvmCtx;
  std::unique_ptr<CodeGenerator> CodeGen;
  clang::CompilerInstance &ci;
  CUDACompiler &Compiler;
  bool Running;

  CustomAstConsumer(llvm::LLVMContext &Ctx,
                    const llvm::StringRef &ModuleName,
                    clang::CompilerInstance &CI,
                    CUDACompiler &Compiler)
      : llvmCtx(Ctx),
        CodeGen{CreateLLVMCodeGen(CI, ModuleName, llvmCtx)}, ci{CI},
        Compiler(Compiler), Running(true) {}

  void Initialize(clang::ASTContext &C) override {
    CodeGen->Initialize(C);
  }
  void HandleCXXStaticMemberVarInstantiation(
      clang::VarDecl *V) override {
    CodeGen->HandleCXXStaticMemberVarInstantiation(V);
  }
  bool HandleTopLevelDecl(clang::DeclGroupRef D) override {
    return CodeGen->HandleTopLevelDecl(D);
  }
  void HandleInlineFunctionDefinition(
      clang::FunctionDecl *D) override {
    CodeGen->HandleInlineFunctionDefinition(D);
  }
  void HandleTagDeclDefinition(clang::TagDecl *D) override {
    CodeGen->HandleTagDeclDefinition(D);
  }
  void HandleTagDeclRequiredDefinition(
      const clang::TagDecl *D) override {
    CodeGen->HandleTagDeclRequiredDefinition(D);
  }
  void AssignInheritanceModel(clang::CXXRecordDecl *D) override {
    CodeGen->AssignInheritanceModel(D);
  }
  void CompleteTentativeDefinition(clang::VarDecl *D) override {
    CodeGen->CompleteTentativeDefinition(D);
  }
  void CompleteExternalDeclaration(
      clang::DeclaratorDecl *D) override {
    CodeGen->CompleteExternalDeclaration(D);
  }
  void HandleVTable(clang::CXXRecordDecl *D) override {
    CodeGen->HandleVTable(D);
  }

  void HandleTranslationUnit(clang::ASTContext &Ctx) override {
    TensorTypeHelpers helper(Ctx, ci.getSema());
    Compiler.CompileExecutionContext->SwitchTo(
        *Compiler.InvocationContext.get());

    while (Running) {
      while (Compiler.TaskQueue.size()) {
        Compiler.TaskQueue.front()(helper, *this);
        Compiler.TaskQueue.pop();
      }
      if (Running)
        Compiler.CompileExecutionContext->SwitchTo(
            *Compiler.InvocationContext.get());
    }
  }
};

struct CustomFEAction : clang::ASTFrontendAction {
  llvm::StringRef ModuleName;
  llvm::LLVMContext &Context;
  CUDACompiler &Compiler;

  CustomFEAction(const llvm::StringRef &ModuleName,
                 llvm::LLVMContext &Ctx, CUDACompiler &Compiler)
      : ModuleName(ModuleName), Context(Ctx), Compiler(Compiler) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef) override {
    return std::make_unique<CustomAstConsumer>(Context, ModuleName, CI,
                                               Compiler);
  }
};

struct LaunchArgs {
  CUDACompiler &Compiler;
  CustomFEAction Action;
  LaunchArgs(CUDACompiler &Compiler, llvm::LLVMContext &Context,
             const llvm::StringRef &ModuleName)
      : Compiler(Compiler), Action(ModuleName, Context, Compiler) {}
};

// ============================================================
// Public API
// ============================================================

std::tuple<std::string, std::string, std::vector<CudaFuncResult>>
tritonCompileCuda(llvm::LLVMContext &ctx, const std::string &source,
                  const std::string &sm, const std::string &resourceDir,
                  const std::vector<std::string> &includePaths,
                  const std::vector<CudaFuncRequest> &requests);

std::string tritonExtractExternCallSpecs(mlir::ModuleOp module);

// D-07: Returns the total number of clang parses performed across all
// compilations in this process. Used for per-compile delta assertions.
int getExternCudaParseCount();

// Patch extern_call op result types based on CUDA-inferred return types.
// Returns empty string on success, error message on failure.
// jsonReturnTypes: {"symbol": [{"scalar":"f32","shape":[512],"reg_basis":[...],...}, ...], ...}
// Array has one entry per result; single-result functions use a one-element array.
std::string
tritonPatchExternCallResultTypes(mlir::ModuleOp module,
                                 const std::string &jsonReturnTypes);

// Links compiled CUDA bitcode into dstMod (same LLVMContext).
void linkBitcodeToModule(llvm::Module *dstMod,
                         const std::string &bitcode,
                         llvm::LLVMContext &ctx);
