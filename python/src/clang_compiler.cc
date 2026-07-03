#include "python/src/clang_compiler.h"

#include <clang/AST/APValue.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/Basic/LangStandard.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Sema/Initialization.h>
#include <clang/Sema/Lookup.h>
#include <clang/Sema/Template.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

#include <filesystem>
#include <fstream>
#include <queue>
#include <sstream>

#ifdef alloca
#undef alloca
#endif

using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::isa;

// ============================================================
// Backend initialization
// ============================================================

static void InitializeNVPTXBackend() {
  LLVMInitializeNVPTXAsmPrinter();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
}

// ============================================================
// x86-64 System V ABI — context switch assembly
// ============================================================

__attribute__((naked)) static void
X64SysVSwitchContext(uint64_t *saveGPR,
                     const uint64_t *restoreGPR) {
  asm volatile(
      "leaq (%%rsp),%%rax\n"
      "movq %%rax, 104(%%rdi)\n"
      "movq %%rbx, 96(%%rdi)\n"
      "movq %%rcx, 88(%%rdi)\n"
      "movq %%rdx, 80(%%rdi)\n"
      "movq 0(%%rax), %%rax\n"
      "movq %%rax, 72(%%rdi)\n"
      "movq %%rsi, 64(%%rdi)\n"
      "movq %%rdi, 56(%%rdi)\n"
      "movq %%rbp, 48(%%rdi)\n"
      "movq %%r8,  40(%%rdi)\n"
      "movq %%r9,  32(%%rdi)\n"
      "movq %%r12, 24(%%rdi)\n"
      "movq %%r13, 16(%%rdi)\n"
      "movq %%r14,  8(%%rdi)\n"
      "movq %%r15, (%%rdi)\n"
      "xorq %%rax, %%rax\n"
      "movq 48(%%rsi), %%rbp\n"
      "movq 104(%%rsi), %%rsp\n"
      "movq (%%rsi), %%r15\n"
      "movq 8(%%rsi), %%r14\n"
      "movq 16(%%rsi), %%r13\n"
      "movq 24(%%rsi), %%r12\n"
      "movq 32(%%rsi), %%r9\n"
      "movq 40(%%rsi), %%r8\n"
      "movq 56(%%rsi), %%rdi\n"
      "movq 80(%%rsi), %%rdx\n"
      "movq 88(%%rsi), %%rcx\n"
      "movq 96(%%rsi), %%rbx\n"
      "leaq 8(%%rsp), %%rsp\n"
      "pushq 72(%%rsi)\n"
      "movq 64(%%rsi), %%rsi\n"
      "ret\n"
      ::
      : "memory", "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rsp",
        "rbp", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
        "cc");
}

const TargetABI X64SysVABI = {
    "x86_64-sysv",
    14,
    16,
    {9, 13, 6, 7, 8},
    X64SysVSwitchContext,
};

// ============================================================
// TypeBuilder — out-of-line definitions
// ============================================================

TypeBuilder::TypeBuilder(clang::ASTContext &Ctx, clang::Sema &S)
    : Ctx(Ctx), SemaRef(S),
      ShapeTemplateType(getTemplateDecl(Ctx, "Shape")),
      LayoutFactoryTemplateType(getTemplateDecl(Ctx, "TensorLayout")),
      IntTupleTemplateType(getTemplateDecl(Ctx, "IntTuple")),
      TensorTemplateType(getTemplateDecl(Ctx, "Tensor")) {}

clang::TemplateArgument TypeBuilder::mkIntegralArgUint32(uint32_t V) {
  return clang::TemplateArgument(Ctx, llvm::APSInt(llvm::APInt(32, V)),
                                 Ctx.UnsignedIntTy);
}

clang::TemplateArgument TypeBuilder::mkTypeArg(clang::QualType T) {
  return clang::TemplateArgument(Ctx.getCanonicalType(T));
}

std::optional<uint32_t> TypeBuilder::EvaluateConstexpr(
    clang::ClassTemplateSpecializationDecl *Spec,
    const llvm::StringRef &Name) {
  auto Rl = Spec->lookup(&Ctx.Idents.get(Name));
  if (Rl.empty())
    return {};
  auto *VD = llvm::cast<clang::VarDecl>(*Rl.begin());
  SemaRef.InstantiateVariableDefinition(VD->getLocation(), VD);
  if (auto *V = VD->evaluateValue()) {
    if (V->isInt())
      return V->getInt().getZExtValue();
  }
  return {};
}

ShapeResult
TypeBuilder::buildShape(llvm::ArrayRef<uint32_t> shapeDims) {
  ShapeResult R;
  auto SL = ShapeTemplateType->getLocation();

  llvm::SmallVector<clang::TemplateArgument, 4> args;
  for (auto d : shapeDims)
    args.push_back(mkIntegralArgUint32(d));
  auto packed = clang::TemplateArgument::CreatePackCopy(Ctx, args);

  void *ins = nullptr;
  if (!(R.spec = ShapeTemplateType->findSpecialization({packed}, ins))) {
    R.spec = clang::ClassTemplateSpecializationDecl::Create(
        Ctx, clang::TagTypeKind::Struct, Ctx.getTranslationUnitDecl(),
        SL, SL, ShapeTemplateType, {packed}, false, nullptr);
    ShapeTemplateType->AddSpecialization(R.spec, ins);
  }
  if (!R.spec->hasDefinition())
    SemaRef.InstantiateClassTemplateSpecialization(
        SL, R.spec, clang::TSK_ImplicitInstantiation, false, false);
  R.type = Ctx.getTemplateSpecializationType(
      clang::ElaboratedTypeKeyword::None,
      clang::TemplateName(ShapeTemplateType), {packed}, {packed},
      Ctx.getCanonicalTagType(R.spec));

  R.dims.RANK = EvaluateConstexpr(R.spec, "RANK").value();
  R.dims.SIZE = EvaluateConstexpr(R.spec, "SIZE").value();
  return R;
}

clang::ClassTemplateSpecializationDecl *
TypeBuilder::BuildIntTuple(clang::SourceLocation SL, unsigned N) {
  auto TypeArg = mkIntegralArgUint32(N);
  void *ins = nullptr;
  clang::ClassTemplateSpecializationDecl *S;
  if (!(S = IntTupleTemplateType->findSpecialization({TypeArg}, ins))) {
    S = clang::ClassTemplateSpecializationDecl::Create(
        Ctx, clang::TagTypeKind::Struct, Ctx.getTranslationUnitDecl(),
        SL, SL, IntTupleTemplateType, {TypeArg}, false, nullptr);
    IntTupleTemplateType->AddSpecialization(S, ins);
  }
  return S;
}

LayoutFactoryContext
TypeBuilder::BuildLayoutFactory(const ShapeResult &shape,
                                uint32_t N_WARPS) {
  LayoutFactoryContext L;
  L.N_WARPS = N_WARPS;

  auto SL = LayoutFactoryTemplateType->getLocation();

  auto *args = clang::TemplateArgumentList::CreateCopy(
      Ctx, {mkTypeArg(shape.type), mkIntegralArgUint32(N_WARPS)});
  void *ins = nullptr;
  if (!(L.spec = LayoutFactoryTemplateType->findSpecialization(
            args->asArray(), ins))) {
    L.spec = clang::ClassTemplateSpecializationDecl::Create(
        Ctx, clang::TagTypeKind::Struct, Ctx.getTranslationUnitDecl(),
        SL, SL, LayoutFactoryTemplateType, args->asArray(), false,
        nullptr);
    LayoutFactoryTemplateType->AddSpecialization(L.spec, ins);
  }
  if (!L.spec->hasDefinition())
    SemaRef.InstantiateClassTemplateSpecialization(
        SL, L.spec, clang::TSK_ImplicitInstantiation, false, false);

  L.BasisGroupTmpl = dyn_cast<clang::ClassTemplateDecl>(
      *L.spec->lookup(&Ctx.Idents.get("BasisGroup")).begin());
  L.LayoutTmpl = dyn_cast<clang::ClassTemplateDecl>(
      *L.spec->lookup(&Ctx.Idents.get("Layout")).begin());

  L.N_LANE_AXES = EvaluateConstexpr(L.spec, "N_LANE_AXES").value();
  L.N_REG_AXES = EvaluateConstexpr(L.spec, "N_REG_AXES").value();
  L.N_WARP_AXES = EvaluateConstexpr(L.spec, "N_WARP_AXES").value();
  return L;
}

std::pair<std::optional<clang::TemplateArgument>,
          clang::ClassTemplateSpecializationDecl *>
TypeBuilder::BuildBasisGroup(const LayoutFactoryContext &LF,
                             unsigned N_BASES,
                             llvm::SmallVector<uint32_t, 4> vecs) {
  assert(!N_BASES || vecs.size() % N_BASES == 0);
  auto SL = LF.BasisGroupTmpl->getLocation();
  using clang::APValue;

  auto NBasesArg = mkIntegralArgUint32(N_BASES);
  void *ins = nullptr;
  clang::ClassTemplateSpecializationDecl *S;
  if (!(S = LF.BasisGroupTmpl->findSpecialization({NBasesArg}, ins))) {
    S = clang::ClassTemplateSpecializationDecl::Create(
        Ctx, clang::TagTypeKind::Struct, LF.spec, SL, SL,
        LF.BasisGroupTmpl, {NBasesArg}, false, nullptr);
    LF.BasisGroupTmpl->AddSpecialization(S, ins);
  }

  clang::QualType BGType = Ctx.getTagType(
      clang::ElaboratedTypeKeyword::None, std::nullopt, S, false);

  llvm::SmallVector<APValue, 4> Elts(N_BASES);
  auto RANK = N_BASES ? vecs.size() / N_BASES : 0;
  for (unsigned i = 0; i < N_BASES; ++i) {
    Elts[i] = APValue(APValue::UninitStruct(), 0u, 1u);
    Elts[i].getStructField(0) =
        APValue(APValue::UninitArray(), RANK, RANK);
    for (unsigned r = 0; r < RANK; ++r)
      Elts[i].getStructField(0).getArrayInitializedElt(r) =
          APValue(
              llvm::APSInt(llvm::APInt(32, vecs[i * RANK + r], false)));
  }
  APValue val(APValue::UninitStruct(), 0u, 1u);
  val.getStructField(0) =
      APValue(APValue::UninitArray(), N_BASES, N_BASES);
  for (unsigned i = 0; i < N_BASES; ++i)
    val.getStructField(0).getArrayInitializedElt(i) =
        std::move(Elts[i]);

  auto *TPOD = Ctx.getTemplateParamObjectDecl(BGType, val);
  return {clang::TemplateArgument(TPOD, Ctx.getCanonicalType(BGType)),
          S};
}

clang::QualType TypeBuilder::BuildLayout(
    const LayoutFactoryContext &LF, clang::TemplateArgument aRegs,
    clang::TemplateArgument aLanes, clang::TemplateArgument aWarps) {
  auto SL = LayoutFactoryTemplateType->getLocation();
  auto *specArgs = clang::TemplateArgumentList::CreateCopy(
      Ctx, {aRegs, aLanes, aWarps});
  void *ins = nullptr;
  clang::ClassTemplateSpecializationDecl *Spec;
  if (!(Spec = LF.LayoutTmpl->findSpecialization(specArgs->asArray(),
                                                  ins))) {
    Spec = clang::ClassTemplateSpecializationDecl::Create(
        Ctx, clang::TagTypeKind::Struct, LF.spec, SL, SL,
        LF.LayoutTmpl, specArgs->asArray(), false, nullptr);
    LF.LayoutTmpl->AddSpecialization(Spec, ins);
  }

  return Ctx.getTemplateSpecializationType(
      clang::ElaboratedTypeKeyword::None,
      clang::TemplateName(LF.LayoutTmpl), specArgs->asArray(),
      specArgs->asArray(), Ctx.getCanonicalTagType(Spec));
}

clang::QualType
TypeBuilder::BuildTensor(clang::QualType ElementType,
                         clang::QualType ShapeType,
                         clang::QualType LayoutType) {
  auto SL = TensorTemplateType->getLocation();
  auto *args = clang::TemplateArgumentList::CreateCopy(
      Ctx, {mkTypeArg(ElementType), mkTypeArg(ShapeType),
            mkTypeArg(LayoutType)});
  void *ins = nullptr;
  clang::ClassTemplateSpecializationDecl *Spec;
  if (!(Spec = TensorTemplateType->findSpecialization(args->asArray(),
                                                       ins))) {
    Spec = clang::ClassTemplateSpecializationDecl::Create(
        Ctx, clang::TagTypeKind::Struct, Ctx.getTranslationUnitDecl(),
        SL, SL, TensorTemplateType, args->asArray(), false, nullptr);
    TensorTemplateType->AddSpecialization(Spec, ins);
  }

  if (!Spec->hasDefinition())
    SemaRef.InstantiateClassTemplateSpecialization(
        SL, Spec, clang::TSK_ImplicitInstantiation, false, false);
  return Ctx.getTemplateSpecializationType(
      clang::ElaboratedTypeKeyword::Struct,
      clang::TemplateName(TensorTemplateType), args->asArray(),
      args->asArray(), Ctx.getCanonicalTagType(Spec));
}

// ============================================================
// TypeInspector — out-of-line definitions
// ============================================================

TypeInspector::TypeInspector(clang::ASTContext &Ctx)
    : Ctx(Ctx), TensorTemplateType(getTemplateDecl(Ctx, "Tensor")) {}

uint32_t TypeInspector::EvaulateConstantTemplateNTTP(
    const clang::TemplateArgument &Arg) {
  if (Arg.getKind() == clang::TemplateArgument::ArgKind::Integral)
    return Arg.getAsIntegral().getZExtValue();
  if (Arg.getKind() == clang::TemplateArgument::ArgKind::Expression) {
    clang::Expr::EvalResult Res;
    Arg.getAsExpr()->EvaluateAsConstantExpr(Res, Ctx);
    return Res.Val.getInt().getZExtValue();
  }
  __builtin_unreachable();
}

llvm::SmallVector<uint32_t, 4>
TypeInspector::ParseShapeType(clang::QualType type) {
  auto *ClassSpecDecl =
      llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
          type->getAsRecordDecl());
  auto TemplatePack =
      ClassSpecDecl->getTemplateArgs().get(0).getPackAsArray();

  auto Dims = TemplatePack.size();
  llvm::SmallVector<uint32_t, 4> Result(Dims);
  for (auto I = 0u; I < Dims; I++)
    Result[I] = EvaulateConstantTemplateNTTP(TemplatePack[I]);
  return Result;
}

llvm::SmallVector<uint32_t, 4>
TypeInspector::ParseBasis(const clang::TemplateArgument &Arg) {
  auto TPOD =
      llvm::dyn_cast<clang::TemplateParamObjectDecl>(Arg.getAsDecl());
  auto &Value = TPOD->getValue().getStructField(0);
  llvm::SmallVector<uint32_t, 8> Result;
  for (auto I = 0u; I < Value.getArraySize(); I++) {
    auto Basis = Value.getArrayInitializedElt(I).getStructField(0);
    for (auto J = 0u; J < Basis.getArraySize(); J++)
      Result.push_back(
          Basis.getArrayInitializedElt(J).getInt().getZExtValue());
  }
  return Result;
}

LayoutInfo TypeInspector::ParseLayoutType(clang::QualType type) {
  auto *ClassSpecDecl =
      llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
          type->getAsRecordDecl());
  auto LayoutFactoryDecl =
      llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
          ClassSpecDecl->getParent());

  auto LayoutShape = ParseShapeType(
      LayoutFactoryDecl->getTemplateArgs().get(0).getAsType());
  auto NumWarps = EvaulateConstantTemplateNTTP(
      LayoutFactoryDecl->getTemplateArgs().get(1));
  auto RegBasis =
      ParseBasis(ClassSpecDecl->getTemplateArgs().get(0));
  auto LaneBasis =
      ParseBasis(ClassSpecDecl->getTemplateArgs().get(1));
  auto WarpBasis =
      ParseBasis(ClassSpecDecl->getTemplateArgs().get(2));

  LayoutInfo info;
  info.LayoutShape.assign(LayoutShape.begin(), LayoutShape.end());
  info.RegBasis.assign(RegBasis.begin(), RegBasis.end());
  info.LaneBasis.assign(LaneBasis.begin(), LaneBasis.end());
  info.WarpBasis.assign(WarpBasis.begin(), WarpBasis.end());
  info.N_WARPS = NumWarps;
  return info;
}

TensorParameter TypeInspector::ParseTensorType(
    clang::ClassTemplateSpecializationDecl *type) {
  auto ScalarType = type->getTemplateArgs().get(0).getAsType();
  auto Shape = ParseShapeType(
      type->getTemplateArgs().get(1).getAsType());
  auto Layout = ParseLayoutType(
      type->getTemplateArgs().get(2).getAsType());

  TensorParameter tp;
  tp.Type = getScalarTypeFromQualType(Ctx, ScalarType);
  tp.Shape.assign(Shape.begin(), Shape.end());
  tp.Layout = Layout;
  return tp;
}

std::variant<std::nullptr_t, TensorParameter>
TypeInspector::DispatchTypeParsing(clang::QualType type) {
  if (auto *RecordDecl = type->getAsRecordDecl()) {
    if (auto *ClassSpecDecl =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                RecordDecl)) {
      if (ClassSpecDecl->getSpecializedTemplate() ==
          TensorTemplateType)
        return ParseTensorType(ClassSpecDecl);
    }
  }
  return nullptr;
}

// ============================================================
// FunctionResolver — out-of-line definitions
// ============================================================

FunctionResolver::FunctionResolver(clang::ASTContext &Ctx,
                                   clang::Sema &S)
    : Ctx(Ctx), SemaRef(S),
      SL(Ctx.getTranslationUnitDecl()->getLocation()) {}

clang::FunctionDecl *FunctionResolver::LookupFunction(
    const llvm::StringRef &Name,
    const llvm::ArrayRef<clang::QualType> &ArgumentTypes) {
  auto R = Ctx.getTranslationUnitDecl()->lookup(
      &Ctx.Idents.get(Name));

  llvm::SmallVector<clang::Expr *, 4> callArgs(
      ArgumentTypes.size());
  for (auto I = 0u; I < ArgumentTypes.size(); I++)
    callArgs[I] = new (Ctx)
        clang::OpaqueValueExpr(SL, ArgumentTypes[I], clang::VK_LValue);

  struct Candidate {
    clang::FunctionDecl *FD;
    clang::FunctionTemplateDecl *FTD;
  };
  std::vector<Candidate> candidates;

  for (auto *D : R) {
    if (auto *FTD =
            llvm::dyn_cast<clang::FunctionTemplateDecl>(D)) {
      clang::sema::TemplateDeductionInfo DI{SL};
      clang::FunctionDecl *cand = nullptr;
      auto Result = SemaRef.DeduceTemplateArguments(
          FTD, nullptr, callArgs, cand, DI, false, false, false,
          clang::QualType(), clang::Expr::Classification(), true,
          [](llvm::ArrayRef<clang::QualType>,
             bool) { return false; });

      if (Result == clang::TemplateDeductionResult::Success && cand)
        candidates.push_back({cand, FTD});
    } else if (auto *FD =
                   llvm::dyn_cast<clang::FunctionDecl>(D)) {
      if (FD->getNumParams() < ArgumentTypes.size())
        continue;

      bool match = true;
      for (auto J = 0u; J < ArgumentTypes.size(); J++) {
        auto *argExpr = new (Ctx) clang::OpaqueValueExpr(
            SL, ArgumentTypes[J], clang::VK_LValue);
        clang::InitializedEntity Entity =
            clang::InitializedEntity::InitializeParameter(
                Ctx, FD->getParamDecl(J)->getType(), false);

        clang::Sema::SFINAETrap trap(SemaRef);
        if (SemaRef
                .PerformCopyInitialization(Entity, SL, argExpr)
                .isInvalid()) {
          match = false;
          break;
        }
      }
      if (match)
        candidates.push_back({FD, nullptr});
    }
  }
  if (candidates.empty())
    return nullptr;

  clang::FunctionDecl *FD = nullptr;
  for (auto &c : candidates) {
    if (!c.FTD) {
      FD = c.FD;
      break;
    }
  }
  if (!FD) {
    auto *bestFTD = std::accumulate(
        candidates.begin() + 1, candidates.end(),
        candidates[0].FTD,
        [&](clang::FunctionTemplateDecl *best, const Candidate &c) {
          auto *r = SemaRef.getMoreSpecializedTemplate(
              best, c.FTD, SL, clang::TPOC_Other, 0);
          return r ? r : best;
        });
    FD = std::find_if(candidates.begin(), candidates.end(),
                      [&](auto &c) { return c.FTD == bestFTD; })
             ->FD;
  }
  return FD;
}

clang::FunctionDecl *
FunctionResolver::InstantiateFunction(clang::FunctionDecl *FD) {
  if (FD->getDescribedFunctionTemplate() || FD->getPrimaryTemplate())
    SemaRef.InstantiateFunctionDefinition(FD->getLocation(), FD,
                                          true);
  return FD;
}

// ============================================================
// CUDACompiler — out-of-line definitions
// ============================================================

void CUDACompiler::PerformCompileImpl(uint64_t Arg0,
                                      ExecutionContext &ExecCtx) {
  clang::noteBottomOfStack(true);
  CUDACompiler *CompilerRef = nullptr;
  do {
    auto Args =
        std::move(*(std::unique_ptr<LaunchArgs> *)Arg0);
    Args->Compiler.CI->ExecuteAction(Args->Action);
    CompilerRef = &Args->Compiler;
  } while (0);

  CompilerRef->CompileExecutionContext->SwitchTo(
      *CompilerRef->InvocationContext);

  __builtin_unreachable();
}

CUDACompiler::CUDACompiler(
    llvm::StringRef SourceCode, int OptLevel, const std::string &sm,
    const std::string &resourceDir,
    const std::vector<std::string> &includePaths)
    : Vfs(llvm::vfs::getRealFileSystem()),
      SourceBuffer{std::move(
          llvm::WritableMemoryBuffer::getNewMemBuffer(
              SourceCode.size()))},
      CI{}, CompileExecutionContext(),
      InvocationContext(
          std::make_unique<ExecutionContext>(X64SysVABI)) {
  std::copy(SourceCode.begin(), SourceCode.end(),
            SourceBuffer->getBufferStart());
  auto inv = std::make_unique<clang::CompilerInvocation>();
  do {
    auto &L = inv->getLangOpts();
    L.CUDA = L.CPlusPlus = L.CPlusPlus11 = L.CPlusPlus14 =
        L.CPlusPlus17 = L.CPlusPlus20 = true;
    L.Bool = L.WChar = L.Char8 = true;
    L.GNUMode = false;
    L.DeclSpecKeyword = true;
    L.GNUAsm = true;
    L.LangStd = clang::LangStandard::lang_cxx20;
    L.CUDAHostDeviceConstexpr = true;
    L.OffloadingNewDriver = true;
    L.DelayedTemplateParsing = true;
    L.Exceptions = L.CXXExceptions = false;
    L.CUDAIsDevice = true;
    L.RoundingMath = false;
    L.ThreadsafeStatics = false;
    L.DebuggerSupport = false;
    L.GNUCVersion = 40201;
    L.ImplicitModules = false;
  } while (0);
  do {
    auto &TargetOpts = inv->getTargetOpts();
    TargetOpts.Triple = "nvptx64-nvidia-cuda";
    TargetOpts.CPU = sm;
    TargetOpts.FeaturesAsWritten.push_back("+ptx88");
  } while (0);
  do {
    auto &FrontendOpts = inv->getFrontendOpts();
    FrontendOpts.ProgramAction =
        clang::frontend::ActionKind::ASTDump;
    FrontendOpts.AuxTriple = "x86_64-unknown-linux-gnu";
    FrontendOpts.AuxTargetCPU = "x86-64";
    FrontendOpts.Inputs.push_back(clang::FrontendInputFile(
        SourceBuffer->getMemBufferRef(),
        clang::InputKind(clang::Language::CUDA)));
  } while (0);
  inv->getDiagnosticOpts().DiagnosticLogFile = "/dev/null";
  do {
    using clang::frontend::IncludeDirGroup;
    auto &H = inv->getHeaderSearchOpts();

    if (!resourceDir.empty()) {
      H.ResourceDir = resourceDir;
      auto resPath = std::filesystem::path(resourceDir);
      H.AddPath((resPath / "include").string(),
                IncludeDirGroup::System, false, true);
      H.AddPath((resPath / "include" / "cuda_wrapper").string(),
                IncludeDirGroup::System, false, true);
    }

    H.AddPath("/usr/include/c++/12", IncludeDirGroup::System, false,
              false);
    H.AddPath("/usr/include/x86_64-linux-gnu/c++/12",
              IncludeDirGroup::System, false, false);
    H.AddPath("/usr/include/c++/12/backward",
              IncludeDirGroup::System, false, false);
    H.AddPath("/usr/include", IncludeDirGroup::System, false, true);
    H.AddPath("/usr/include/x86_64-linux-gnu",
              IncludeDirGroup::System, false, true);

    for (const auto &p : includePaths)
      H.AddPath(p, IncludeDirGroup::System, false, true);
  } while (0);
  do {
    auto &PPOpts = inv->getPreprocessorOpts();
    PPOpts.Includes.push_back("__clang_cuda_runtime_wrapper.h");
  } while (0);
  do {
    auto &CGOpts = inv->getCodeGenOpts();
    CGOpts.Autolink = false;
    CGOpts.RelocationModel = llvm::Reloc::Static;
    CGOpts.VectorizeSLP = true;
    CGOpts.OptimizationLevel = OptLevel;
  } while (0);
  CI = std::make_unique<clang::CompilerInstance>(std::move(inv));
  CI->createDiagnostics();
  CI->setVirtualFileSystem(Vfs);
  CI->createFileManager();
  CI->createSourceManager();
}

void CUDACompiler::PerformParse(
    llvm::LLVMContext &Context,
    const llvm::StringRef &ModuleName) {
  auto launchArgs =
      std::make_unique<LaunchArgs>(*this, Context, ModuleName);
  CompileExecutionContext =
      std::make_unique<ExecutionContext>(
          X64SysVABI, PerformCompileImpl, nullptr,
          (uint64_t)(&launchArgs), clang::DesiredStackSize);
  InvocationContext->SwitchTo(*CompileExecutionContext);
}

clang::QualType
CUDACompiler::BuildTensor(const TensorParameter &Param) {
  clang::QualType Result;
  TaskQueue.emplace([&](TensorTypeHelpers &helper,
                        CustomAstConsumer &) {
    auto Shape = helper.Builder.buildShape(Param.Shape);
    auto &LayoutInfo = Param.Layout;
    auto LayoutShape =
        helper.Builder.buildShape(LayoutInfo.LayoutShape);

    auto LayoutFactory = helper.Builder.BuildLayoutFactory(
        LayoutShape, LayoutInfo.N_WARPS);
    auto RegBasis = helper.Builder.BuildBasisGroup(
        LayoutFactory, LayoutFactory.N_REG_AXES,
        llvm::SmallVector<uint32_t, 4>(LayoutInfo.RegBasis.begin(),
                                        LayoutInfo.RegBasis.end()));
    auto LaneBasis = helper.Builder.BuildBasisGroup(
        LayoutFactory, LayoutFactory.N_LANE_AXES,
        llvm::SmallVector<uint32_t, 4>(
            LayoutInfo.LaneBasis.begin(), LayoutInfo.LaneBasis.end()));
    auto WarpBasis = helper.Builder.BuildBasisGroup(
        LayoutFactory, LayoutFactory.N_WARP_AXES,
        llvm::SmallVector<uint32_t, 4>(
            LayoutInfo.WarpBasis.begin(), LayoutInfo.WarpBasis.end()));

    auto Layout = helper.Builder.BuildLayout(
        LayoutFactory, RegBasis.first.value(),
        LaneBasis.first.value(), WarpBasis.first.value());
    Result = helper.Builder.BuildTensor(
        getQualTypeFromScalarType(helper.Builder.Ctx, Param.Type),
        Shape.type, Layout);
  });
  InvocationContext->SwitchTo(*CompileExecutionContext);
  return Result;
}

clang::FunctionDecl *CUDACompiler::LookupFunction(
    const llvm::StringRef &Name,
    const llvm::ArrayRef<clang::QualType> &Args) {
  clang::FunctionDecl *Result;
  TaskQueue.emplace([&](TensorTypeHelpers &helper,
                        CustomAstConsumer &) {
    Result = helper.Resolver.LookupFunction(Name, Args);
  });
  InvocationContext->SwitchTo(*CompileExecutionContext);
  return Result;
}

llvm::Function *
CUDACompiler::InstantiationFunction(clang::FunctionDecl *FD) {
  llvm::Function *Result;
  TaskQueue.emplace([&](TensorTypeHelpers &helper,
                        CustomAstConsumer &AstC) {
    helper.Resolver.InstantiateFunction(FD);
    if (FD->isFunctionTemplateSpecialization())
      FD->setTemplateSpecializationKind(
          clang::TSK_ExplicitInstantiationDefinition);
    AstC.CodeGen->HandleTopLevelDecl(clang::DeclGroupRef(FD));
    Result = llvm::cast<llvm::Function>(
        AstC.CodeGen->GetAddrOfGlobal(FD, true));
  });
  InvocationContext->SwitchTo(*CompileExecutionContext);
  return Result;
}

std::variant<std::nullptr_t, TensorParameter>
CUDACompiler::EvaluateFunctionReturnType(
    clang::FunctionDecl *FD) {
  std::variant<std::nullptr_t, TensorParameter> Result = nullptr;
  TaskQueue.emplace([&](TensorTypeHelpers &helper,
                        CustomAstConsumer &) {
    Result = helper.Inspector.DispatchTypeParsing(
        FD->getCallResultType().getCanonicalType());
  });
  InvocationContext->SwitchTo(*CompileExecutionContext);
  return Result;
}

std::unique_ptr<llvm::Module> CUDACompiler::EmitFinalModule() {
  std::unique_ptr<llvm::Module> Result;
  TaskQueue.emplace([&](TensorTypeHelpers &helper,
                        CustomAstConsumer &AstC) {
    AstC.CodeGen->HandleTranslationUnit(
        AstC.ci.getASTContext());
    AstC.Running = false;
    Result = AstC.CodeGen->ReleaseModule();
  });
  InvocationContext->SwitchTo(*CompileExecutionContext);
  return Result;
}

// ============================================================
// MLIR spec extraction — scan MLIR module for ttg.extern_call ops
// ============================================================
namespace {

struct SpecInput {
  std::string dtype;
  llvm::SmallVector<int64_t, 4> shape;
  int64_t numWarps;
  llvm::SmallVector<int32_t, 16> regBases;
  llvm::SmallVector<int32_t, 16> laneBases;
  llvm::SmallVector<int32_t, 16> warpBases;
};

struct ExternCallSpec {
  std::string symbol;
  std::string libpath;
  bool useFastMath = false;
  llvm::SmallVector<SpecInput, 4> inputs;
};

llvm::SmallVector<ExternCallSpec, 4>
extractExternCallSpecs(mlir::ModuleOp module) {
  using namespace mlir;
  llvm::SmallVector<ExternCallSpec, 4> results;

  StringAttr kRegister =
      StringAttr::get(module.getContext(), "register");
  StringAttr kLane =
      StringAttr::get(module.getContext(), "lane");
  StringAttr kWarp =
      StringAttr::get(module.getContext(), "warp");

  module.walk([&](mlir::triton::gpu::ExternCallOp op) {
    ExternCallSpec spec;
    spec.symbol = op.getSymbol().str();
    spec.libpath = op.getLibpath().str();
    spec.useFastMath = op.getUseFastMath();

    for (auto operand : op.getInputs()) {
      auto tensorTy = cast<RankedTensorType>(operand.getType());
      auto shape = tensorTy.getShape();
      auto encoding = tensorTy.getEncoding();

      auto ll =
          mlir::triton::gpu::toLinearLayout(shape, encoding);

      SpecInput input;
      input.shape.assign(shape.begin(), shape.end());
      input.numWarps = ll.getInDimSize(kWarp);

      auto flattenBases = [](auto bases) {
        llvm::SmallVector<int32_t, 16> flat;
        for (auto &row : bases)
          flat.append(row.begin(), row.end());
        return flat;
      };

      input.regBases =
          flattenBases(ll.getBases().lookup(kRegister));
      input.laneBases =
          flattenBases(ll.getBases().lookup(kLane));
      input.warpBases =
          flattenBases(ll.getBases().lookup(kWarp));

      auto elemTy = tensorTy.getElementType();
      if (isa<Float32Type>(elemTy))
        input.dtype = "f32";
      else if (isa<Float64Type>(elemTy))
        input.dtype = "f64";
      else if (isa<Float16Type>(elemTy))
        input.dtype = "f16";
      else if (isa<BFloat16Type>(elemTy))
        input.dtype = "bf16";
      else if (elemTy.isInteger(32))
        input.dtype = "i32";
      else if (elemTy.isInteger(64))
        input.dtype = "i64";
      else if (elemTy.isInteger(8))
        input.dtype = "i8";
      else
        input.dtype = "f32";

      spec.inputs.push_back(std::move(input));
    }
    results.push_back(std::move(spec));
  });

  return results;
}

} // namespace

// ============================================================
// Public API
// ============================================================

std::string
tritonExtractExternCallSpecs(mlir::ModuleOp module) {
  auto specs = extractExternCallSpecs(module);

  std::string jsonStr;
  llvm::raw_string_ostream os(jsonStr);
  os << "[";
  bool firstSpec = true;
  for (auto &spec : specs) {
    if (!firstSpec)
      os << ", ";
    firstSpec = false;
    os << "{";
    os << "\"symbol\": \"" << spec.symbol << "\", ";
    os << "\"libpath\": \"" << spec.libpath << "\", ";
    os << "\"use_fast_math\": " << (spec.useFastMath ? "true" : "false") << ", ";
    os << "\"inputs\": [";
    bool firstInput = true;
    for (auto &input : spec.inputs) {
      if (!firstInput)
        os << ", ";
      firstInput = false;
      os << "{";
      os << "\"dtype\": \"" << input.dtype << "\", ";
      os << "\"shape\": [";
      for (size_t i = 0; i < input.shape.size(); ++i) {
        if (i > 0)
          os << ", ";
        os << input.shape[i];
      }
      os << "], ";
      os << "\"num_warps\": " << input.numWarps << ", ";
      auto flatten = [&](auto &bases) {
        os << "[";
        for (size_t i = 0; i < bases.size(); ++i) {
          if (i > 0)
            os << ", ";
          os << bases[i];
        }
        os << "]";
      };
      os << "\"reg_bases\": ";
      flatten(input.regBases);
      os << ", \"lane_bases\": ";
      flatten(input.laneBases);
      os << ", \"warp_bases\": ";
      flatten(input.warpBases);
      os << "}";
    }
    os << "]}";
  }
  os << "]";
  os.flush();
  return jsonStr;
}

std::string tritonPatchExternCallResultTypes(
    mlir::ModuleOp module, const std::string &jsonReturnTypes) {
  auto jsonVal = llvm::json::parse(jsonReturnTypes);
  if (!jsonVal)
    return "Failed to parse return type JSON";

  auto *jsonObj = jsonVal->getAsObject();
  if (!jsonObj)
    return "Return type JSON is not an object";

  struct InferredType {
    std::string scalar;
    std::vector<int64_t> shape;
    std::vector<uint32_t> layoutShape;
    std::vector<uint32_t> regBasis;
    std::vector<uint32_t> laneBasis;
    std::vector<uint32_t> warpBasis;
    uint32_t nWarps;
  };
  llvm::StringMap<InferredType> inferredMap;

  for (auto &kv : *jsonObj) {
    auto *obj = kv.second.getAsObject();
    if (!obj)
      continue;

    InferredType info;
    if (auto s = obj->getString("scalar"))
      info.scalar = s->str();
    if (auto arr = obj->getArray("shape"))
      for (auto &v : *arr)
        if (auto i = v.getAsInteger())
          info.shape.push_back(*i);
    if (auto arr = obj->getArray("layout_shape"))
      for (auto &v : *arr)
        if (auto i = v.getAsInteger())
          info.layoutShape.push_back(*i);
    if (auto arr = obj->getArray("reg_basis"))
      for (auto &v : *arr)
        if (auto i = v.getAsInteger())
          info.regBasis.push_back(static_cast<uint32_t>(*i));
    if (auto arr = obj->getArray("lane_basis"))
      for (auto &v : *arr)
        if (auto i = v.getAsInteger())
          info.laneBasis.push_back(static_cast<uint32_t>(*i));
    if (auto arr = obj->getArray("warp_basis"))
      for (auto &v : *arr)
        if (auto i = v.getAsInteger())
          info.warpBasis.push_back(static_cast<uint32_t>(*i));
    if (auto n = obj->getInteger("n_warps"))
      info.nWarps = static_cast<uint32_t>(*n);

    inferredMap[kv.first] = std::move(info);
  }

  if (inferredMap.empty())
    return "";

  using namespace mlir;
  auto *ctx = module.getContext();
  StringAttr kReg = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");

  auto getElemType = [&](const std::string &name) -> Type {
    if (name == "f32")  return Float32Type::get(ctx);
    if (name == "f16")  return Float16Type::get(ctx);
    if (name == "bf16") return BFloat16Type::get(ctx);
    if (name == "i32")  return IntegerType::get(ctx, 32);
    if (name == "i64")  return IntegerType::get(ctx, 64);
    return Float32Type::get(ctx);
  };

  auto buildEncodedType =
      [&](const InferredType &info, Type elemTy) -> RankedTensorType {
    auto rank = info.layoutShape.size();
    auto unflatten = [&](const std::vector<uint32_t> &flat) {
      size_t n = rank ? flat.size() / rank : 0;
      std::vector<std::vector<int32_t>> result(n);
      for (size_t i = 0; i < n; i++)
        for (size_t r = 0; r < rank; r++)
          result[i].push_back(
              static_cast<int32_t>(flat[i * rank + r]));
      return result;
    };
    auto regBases = unflatten(info.regBasis);
    auto laneBases = unflatten(info.laneBasis);
    auto warpBases = unflatten(info.warpBasis);

    auto outDims = triton::standardOutDimPairs(
        ctx, llvm::SmallVector<int64_t>(info.shape.begin(),
                                         info.shape.end()));
    triton::LinearLayout inferredLayout(
        {{kReg, regBases},
         {kLane, laneBases},
         {kWarp, warpBases},
         {kBlock, std::vector<std::vector<int32_t>>{}}},
        outDims,
        /*requiresSurjective=*/true);
    auto inferredEncoding =
        triton::gpu::LinearEncodingAttr::get(ctx, inferredLayout);
    return RankedTensorType::get(
        llvm::SmallVector<int64_t>(info.shape.begin(), info.shape.end()),
        elemTy, inferredEncoding);
  };

  llvm::SmallVector<triton::gpu::ExternCallOp, 4> opsToPatch;
  module.walk(
      [&](triton::gpu::ExternCallOp op) { opsToPatch.push_back(op); });

  for (auto op : opsToPatch) {
    auto symbol = op.getSymbol().str();
    auto it = inferredMap.find(symbol);
    if (it == inferredMap.end())
      continue;

    auto &info = it->second;
    auto declaredType =
        mlir::cast<RankedTensorType>(op.getResult(0).getType());

    auto elemTy = getElemType(info.scalar);
    auto declaredElemTy = declaredType.getElementType();
    auto declaredShape = declaredType.getShape();

    if (elemTy != declaredElemTy)
      return "gl.call: return dtype mismatch for '" + symbol +
             "': CUDA returns " + info.scalar +
             " but declared type has different element type";

    auto shapeRef = llvm::ArrayRef<int64_t>(info.shape);
    if (shapeRef != declaredShape)
      return "gl.call: return shape mismatch for '" + symbol +
             "': CUDA returns different shape than declared";

    auto inferredType = buildEncodedType(info, elemTy);
    if (inferredType == declaredType)
      continue;

    if (op.getAssertNoConv())
      return "gl.call: layout mismatch for '" + symbol +
             "' but assert_no_conv=True";

    OpBuilder builder(op);
    ImplicitLocOpBuilder ib(op.getLoc(), builder);
    auto newOp = triton::gpu::ExternCallOp::create(
        ib, mlir::TypeRange{inferredType}, op.getInputs(),
        op.getSymbol(), op.getLibpath(), op.getAssertNoConv());
    auto convert = triton::gpu::ConvertLayoutOp::create(
        ib, declaredType, newOp.getResult(0));
    op.getResult(0).replaceAllUsesExcept(convert.getResult(), convert);
    op.erase();
  }

  return "";
}

std::tuple<std::string, std::string, std::vector<CudaFuncResult>>
tritonCompileCuda(llvm::LLVMContext &ctx, const std::string &source,
                  const std::string &sm,
                  const std::string &resourceDir,
                  const std::vector<std::string> &includePaths,
                  const std::vector<CudaFuncRequest> &requests) {
  std::vector<CudaFuncResult> results;
  InitializeNVPTXBackend();

  // Check for conflicting use_fast_math hints on the same symbol.
  llvm::StringMap<bool> fastMathMap;
  for (auto &req : requests) {
    auto it = fastMathMap.find(req.Symbol);
    if (it != fastMathMap.end() && it->second != req.UseFastMath)
      return {"",
              "conflicting use_fast_math hints for '" + req.Symbol + "'",
              {}};
    fastMathMap[req.Symbol] = req.UseFastMath;
  }

  CUDACompiler compiler(source, 3, sm, resourceDir, includePaths);
  compiler.PerformParse(ctx, "cudamod");

  std::vector<clang::FunctionDecl *> resolvedFDs;
  std::vector<std::optional<TensorParameter>> returnTypes;

  // Phase 1: Type inference for all calls
  for (auto &req : requests) {
    llvm::SmallVector<clang::QualType, 4> argTypes(
        req.ParamTypes.size());

    for (size_t J = 0; J < req.ParamTypes.size(); J++) {
      if (auto *tp =
              std::get_if<TensorParameter>(&req.ParamTypes[J])) {
        argTypes[J] = compiler.BuildTensor(*tp);
      } else if (auto *st = std::get_if<ScalarType>(
                     &req.ParamTypes[J])) {
        clang::QualType Result;
        compiler.TaskQueue.emplace(
            [&, st](TensorTypeHelpers &helper,
                    CustomAstConsumer &) {
              Result = getQualTypeFromScalarType(helper.Builder.Ctx,
                                                 *st);
            });
        compiler.InvocationContext->SwitchTo(
            *compiler.CompileExecutionContext);
        argTypes[J] = Result;
      }
    }

    auto *FD =
        compiler.LookupFunction(req.Symbol, argTypes);
    if (!FD)
      return {"", "Function lookup failed: " + req.Symbol, {}};

    auto ret = compiler.EvaluateFunctionReturnType(FD);
    std::optional<TensorParameter> retParam;
    if (auto *tp = std::get_if<TensorParameter>(&ret))
      retParam = std::move(*tp);

    resolvedFDs.push_back(FD);
    returnTypes.push_back(std::move(retParam));
  }

  // Phase 2: Codegen for all resolved functions
  for (size_t i = 0; i < resolvedFDs.size(); ++i) {
    auto *fn = compiler.InstantiationFunction(resolvedFDs[i]);
    if (!fn)
      return {"", "Instantiation failed for " + requests[i].Symbol,
              {}};

    if (requests[i].UseFastMath) {
      for (auto &BB : *fn)
        for (auto &I : BB)
          if (isa<llvm::FPMathOperator>(I))
            I.setFast(true);
    }

    CudaFuncResult r;
    r.Symbol = requests[i].Symbol;
    r.MangledName = fn->getName().str();
    r.ReturnType = std::move(returnTypes[i]);
    results.push_back(std::move(r));
  }

  // Phase 3: Finalize
  auto mod = compiler.EmitFinalModule();
  if (!mod)
    return {"", "Failed to emit LLVM module", {}};

  std::string bitcode;
  {
    llvm::raw_string_ostream os(bitcode);
    llvm::WriteBitcodeToFile(*mod, os);
  }

  return {bitcode, "", results};
}

void linkBitcodeToModule(llvm::Module *dstMod,
                         const std::string &bitcode,
                         llvm::LLVMContext &ctx) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(bitcode.data(), bitcode.size()));

  llvm::SMDiagnostic parseErr;
  auto tmpMod = llvm::parseIR(*buf, parseErr, ctx);
  if (!tmpMod) {
    throw std::invalid_argument(
        "Failed to parse CUDA bitcode: " +
        parseErr.getMessage().str());
  }

  llvm::SmallVector<llvm::Function *, 2> srcDefs;
  for (llvm::Function &F : tmpMod->functions())
    if (!F.isDeclaration())
      srcDefs.push_back(&F);

  for (llvm::Function *srcFn : srcDefs) {
    llvm::Function *dstFn =
        dstMod->getFunction(srcFn->getName());
    if (!dstFn || !dstFn->isDeclaration())
      continue;

    llvm::ValueToValueMapTy vmap;
    auto dstArgIt = dstFn->arg_begin();
    for (auto &srcArg : srcFn->args()) {
      if (dstArgIt == dstFn->arg_end())
        break;
      vmap[&srcArg] = &*dstArgIt;
      dstArgIt->setName(srcArg.getName());
      ++dstArgIt;
    }

    llvm::SmallVector<llvm::ReturnInst *, 8> returns;
    llvm::CloneFunctionInto(
        dstFn, srcFn, vmap,
        llvm::CloneFunctionChangeType::DifferentModule, returns);

    for (auto &BB : *dstFn) {
      for (auto &I : BB) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&I);
        if (!call)
          continue;
        auto *callee = call->getCalledFunction();
        if (!callee || callee->getParent() == dstMod)
          continue;
        auto *localFn =
            dstMod->getFunction(callee->getName());
        if (!localFn) {
          localFn = llvm::Function::Create(
              callee->getFunctionType(), callee->getLinkage(),
              callee->getName(), dstMod);
        }
        call->setCalledFunction(localFn);
      }
    }

    srcFn->eraseFromParent();

    for (auto &BB : *dstFn) {
      for (auto RI = BB.begin(); RI != BB.end();) {
        auto &I = *RI++;
        auto *ret = llvm::dyn_cast<llvm::ReturnInst>(&I);
        if (!ret || !ret->getReturnValue())
          continue;
        auto *retVal = ret->getReturnValue();
        auto *retTy = retVal->getType();
        auto *funcRetTy = dstFn->getReturnType();
        if (retTy == funcRetTy)
          continue;
        auto *srcST = llvm::cast<llvm::StructType>(retTy);
        auto *dstST =
            llvm::cast<llvm::StructType>(funcRetTy);
        llvm::Value *newVal =
            llvm::UndefValue::get(funcRetTy);
        for (unsigned i = 0; i < srcST->getNumElements(); ++i) {
          auto *field = llvm::ExtractValueInst::Create(
              retVal, {i}, "", ret->getIterator());
          auto *fieldTy = field->getType();
          auto *dstFieldTy = dstST->getElementType(i);
          llvm::Value *castField = field;
          if (fieldTy != dstFieldTy) {
            auto *slot = new llvm::AllocaInst(
                dstFieldTy, 0, "", ret->getIterator());
            new llvm::StoreInst(field, slot,
                                ret->getIterator());
            castField = new llvm::LoadInst(
                dstFieldTy, slot, "", ret->getIterator());
          }
          newVal = llvm::InsertValueInst::Create(
              newVal, castField, {i}, "", ret->getIterator());
        }
        ret->setOperand(0, newVal);
      }
    }

    if (auto *sub = dstFn->getSubprogram()) {
      if (!sub->isDistinct()) {
        auto &llCtx = dstFn->getContext();
        auto *newSub = llvm::DISubprogram::getDistinct(
            llCtx, sub->getScope(), sub->getName(),
            sub->getLinkageName(), sub->getFile(), sub->getLine(),
            sub->getType(), sub->getScopeLine(),
            sub->getContainingType(), sub->getVirtualIndex(),
            sub->getThisAdjustment(), sub->getFlags(),
            sub->getSPFlags(), sub->getUnit());
        dstFn->setSubprogram(newSub);
      }
    }

    dstFn->addFnAttr(llvm::Attribute::AlwaysInline);
  }

  llvm::StripDebugInfo(*dstMod);
}
