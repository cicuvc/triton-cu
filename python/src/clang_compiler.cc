#include "python/src/clang_compiler.h"

#include <fstream>
#include <sstream>
#include <functional>
#include <filesystem>
#include <variant>
#include <numeric>
#include <unordered_set>

#include <clang/AST/APValue.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclBase.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/DeclarationName.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/GlobalDecl.h>
#include <clang/AST/NestedNameSpecifierBase.h>
#include <clang/AST/PrettyPrinter.h>
#include <clang/AST/TemplateBase.h>
#include <clang/AST/TemplateName.h>
#include <clang/AST/TypeBase.h>
#include <clang/AST/AST.h>
#include <clang/Basic/IdentifierTable.h>
#include <clang/Basic/LangStandard.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Sema/Sema.h>
#include <clang/Sema/Initialization.h>
#include <clang/Sema/Lookup.h>
#include <clang/Sema/Template.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/DebugInfo.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

#ifdef alloca
#undef alloca
#endif

using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::isa;

// ============================================================================
// Internal structures for TensorTypeHelpers
// ============================================================================

struct Dims {
    unsigned RANK = 0;
    uint64_t SIZE = 1;
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

// ============================================================================
// Utility: NVIDIA target initialization (NVPTX only, no AMDGPU)
// ============================================================================

static void InitializeNVPTXBackend() {
    LLVMInitializeNVPTXAsmPrinter();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
}

// ============================================================================
// ManualCompleteDefinition — completes a class template specialization
// ============================================================================

static void ManualCompleteDefinition(clang::ASTContext &Ctx, clang::Sema &S,
                                     clang::SourceLocation SL,
                                     clang::ClassTemplateSpecializationDecl *CTSD) {
    if (CTSD->hasDefinition())
        return;
    CTSD->startDefinition();
    auto *Pattern = CTSD->getSpecializedTemplate()->getTemplatedDecl();
    auto Args = S.getTemplateInstantiationArgs(CTSD);
    for (auto *Member : Pattern->decls()) {
        if (!isa<clang::FieldDecl>(Member))
            continue;
        auto *NewField =
            cast_or_null<clang::FieldDecl>(S.SubstDecl(Member, CTSD, Args));
        if (NewField) {
            NewField->setAccess(clang::AS_public);
            CTSD->addDecl(NewField);
        }
    }
    CTSD->setCompleteDefinition(true);
}

// ============================================================================
// TensorTypeHelpers — build Triton tensor types and instantiate functions
// ============================================================================

namespace {

clang::QualType getQualTypeFromScalarType(clang::ASTContext &Ctx,
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

struct TensorTypeHelpers {
    clang::ASTContext &Ctx;
    clang::Sema &SemaRef;

    clang::ClassTemplateDecl *ShapeTemplateType;
    clang::ClassTemplateDecl *LayoutFactoryTemplateType;
    clang::ClassTemplateDecl *IntTupleTemplateType;
    clang::ClassTemplateDecl *TensorTemplateType;

    clang::ClassTemplateDecl *getTemplateDecl(const char *Name) {
        auto R = Ctx.getTranslationUnitDecl()->lookup(&Ctx.Idents.get(Name));
        if (R.empty())
            return nullptr;
        return dyn_cast<clang::ClassTemplateDecl>(*R.begin());
    }
    clang::TemplateArgument mkIntegralArgUint32(uint32_t V) {
        return clang::TemplateArgument(Ctx, llvm::APSInt(llvm::APInt(32, V)),
                                       Ctx.UnsignedIntTy);
    }
    clang::TemplateArgument mkTypeArg(clang::QualType T) {
        return clang::TemplateArgument(Ctx.getCanonicalType(T));
    }
    std::optional<uint32_t>
    EvaluateConstexpr(clang::ClassTemplateSpecializationDecl *Spec,
                      const llvm::StringRef &Name) {
        auto Rl = Spec->lookup(&Ctx.Idents.get(Name));
        if (Rl.empty())
            return {};
        auto *VD = cast<clang::VarDecl>(*Rl.begin());
        SemaRef.InstantiateVariableDefinition(VD->getLocation(), VD);
        if (auto *V = VD->evaluateValue()) {
            if (V->isInt())
                return V->getInt().getZExtValue();
        }
        return {};
    }

    TensorTypeHelpers(clang::ASTContext &Ctx, clang::Sema &SemaRef)
        : Ctx(Ctx), SemaRef(SemaRef) {
        ShapeTemplateType = getTemplateDecl("Shape");
        LayoutFactoryTemplateType = getTemplateDecl("TensorLayout");
        IntTupleTemplateType = getTemplateDecl("IntTuple");
        TensorTemplateType = getTemplateDecl("Tensor");
    }

    ShapeResult buildShape(llvm::ArrayRef<uint32_t> shapeDims) {
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
    BuildIntTuple(clang::SourceLocation SL, unsigned N) {
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

    LayoutFactoryContext BuildLayoutFactory(const ShapeResult &shape,
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
    BuildBasisGroup(const LayoutFactoryContext &LF, unsigned N_BASES,
                    llvm::ArrayRef<uint32_t> vecs) {
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
            for (unsigned r = 0; r < RANK; ++r) {
                Elts[i].getStructField(0).getArrayInitializedElt(r) =
                    APValue(
                        llvm::APSInt(llvm::APInt(32, vecs[i * RANK + r], false)));
            }
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

    clang::QualType BuildLayout(const LayoutFactoryContext &LF,
                                clang::TemplateArgument aRegs,
                                clang::TemplateArgument aLanes,
                                clang::TemplateArgument aWarps) {
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

    clang::QualType BuildTensor(clang::QualType ElementType,
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

    clang::FunctionDecl *InstantiateFunction(
        const llvm::StringRef &Name,
        const llvm::SmallVector<clang::QualType, 4> &ArgumentTypes) {
        auto *TU = Ctx.getTranslationUnitDecl();
        auto R = TU->lookup(&Ctx.Idents.get(Name));
        auto SL = LayoutFactoryTemplateType->getLocation();

        llvm::SmallVector<clang::Expr *, 4> callArgs(ArgumentTypes.size());
        for (auto I = 0; I < ArgumentTypes.size(); I++) {
            callArgs[I] = new (Ctx) clang::OpaqueValueExpr(
                SL, ArgumentTypes[I], clang::VK_LValue);
        }

        struct Candidate {
            clang::FunctionDecl *FD;
            clang::FunctionTemplateDecl *FTD;
        };
        std::vector<Candidate> candidates;

        for (auto *D : R) {
            if (auto *FTD = dyn_cast<clang::FunctionTemplateDecl>(D)) {
                clang::sema::TemplateDeductionInfo DI{SL};
                clang::FunctionDecl *cand = nullptr;
                auto Result = SemaRef.DeduceTemplateArguments(
                    FTD, /*ExplicitTemplateArgs=*/nullptr, callArgs, cand, DI,
                    /*PartialOverloading=*/false,
                    /*AggregateDeductionCandidate=*/false,
                    /*PartialOrdering=*/false,
                    /*ObjectType=*/clang::QualType(),
                    /*ObjectClassification=*/clang::Expr::Classification(),
                    /*ForOverloadSetAddressResolution=*/true,
                    /*CheckNonDependent=*/
                    [](llvm::ArrayRef<clang::QualType> z, bool b) {
                        return false;
                    });

                if ((Result == clang::TemplateDeductionResult::Success) && cand) {
                    candidates.push_back({cand, FTD});
                }
            } else if (auto *FD = dyn_cast<clang::FunctionDecl>(D)) {
                if (FD->getNumParams() < ArgumentTypes.size())
                    continue;

                bool match = true;
                for (auto J = 0; J < ArgumentTypes.size(); J++) {
                    auto argTy = ArgumentTypes[J];
                    auto paramTy = FD->getParamDecl(J)->getType();

                    auto *argExpr = new (Ctx) clang::OpaqueValueExpr(
                        SL, argTy, clang::VK_LValue);
                    clang::InitializedEntity Entity =
                        clang::InitializedEntity::InitializeParameter(
                            Ctx, paramTy, /*consumed=*/false);

                    clang::Sema::SFINAETrap trap(SemaRef);
                    clang::ExprResult res =
                        SemaRef.PerformCopyInitialization(Entity, SL, argExpr);
                    if (res.isInvalid()) {
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
                candidates.begin() + 1, candidates.end(), candidates[0].FTD,
                [&](clang::FunctionTemplateDecl *best, const Candidate &c) {
                    auto *r = SemaRef.getMoreSpecializedTemplate(
                        best, c.FTD, SL, clang::TPOC_Other, 0);
                    return r ? r : best;
                });
            auto it =
                std::find_if(candidates.begin(), candidates.end(),
                             [&](auto &c) { return c.FTD == bestFTD; });
            FD = it->FD;
        }

        if (FD->getDescribedFunctionTemplate() || FD->getPrimaryTemplate())
            SemaRef.InstantiateFunctionDefinition(SL, FD, true);
        return FD;
    }
};

} // namespace

// ============================================================================
// CustomAstConsumer — clang AST consumer building LLVM IR
// ============================================================================

struct CustomAstConsumer : clang::CodeGenerator {
    llvm::LLVMContext &llvmCtx;
    std::unique_ptr<CodeGenerator> CodeGen;
    clang::CompilerInstance &ci;
    std::function<void(std::unique_ptr<llvm::Module>)> ModuleGenCallback;
    llvm::SmallVector<DeviceFunctionInstantiation, 4> &Instantiations;

    CustomAstConsumer(
        llvm::LLVMContext &Ctx, const llvm::StringRef &ModuleName,
        clang::CompilerInstance &CI,
        std::function<void(std::unique_ptr<llvm::Module>)> &ModuleGenCallback,
        llvm::SmallVector<DeviceFunctionInstantiation, 4> &InstFunc)
        : llvmCtx(Ctx),
          CodeGen{CreateLLVMCodeGen(CI, ModuleName, llvmCtx)},
          ci{CI},
          ModuleGenCallback{ModuleGenCallback},
          Instantiations(InstFunc) {}

    void Initialize(clang::ASTContext &C) override { CodeGen->Initialize(C); }
    void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *V) override {
        CodeGen->HandleCXXStaticMemberVarInstantiation(V);
    }
    bool HandleTopLevelDecl(clang::DeclGroupRef D) override {
        return CodeGen->HandleTopLevelDecl(D);
    }
    void HandleInlineFunctionDefinition(clang::FunctionDecl *D) override {
        CodeGen->HandleInlineFunctionDefinition(D);
    }
    void HandleTagDeclDefinition(clang::TagDecl *D) override {
        CodeGen->HandleTagDeclDefinition(D);
    }
    void HandleTagDeclRequiredDefinition(const clang::TagDecl *D) override {
        CodeGen->HandleTagDeclRequiredDefinition(D);
    }
    void AssignInheritanceModel(clang::CXXRecordDecl *D) override {
        CodeGen->AssignInheritanceModel(D);
    }
    void CompleteTentativeDefinition(clang::VarDecl *D) override {
        CodeGen->CompleteTentativeDefinition(D);
    }
    void CompleteExternalDeclaration(clang::DeclaratorDecl *D) override {
        CodeGen->CompleteExternalDeclaration(D);
    }
    void HandleVTable(clang::CXXRecordDecl *D) override {
        CodeGen->HandleVTable(D);
    }

    void HandleTranslationUnit(clang::ASTContext &Ctx) override {
        TensorTypeHelpers helper(Ctx, ci.getSema());
        for (auto &I : Instantiations) {
            llvm::SmallVector<clang::QualType, 4> ArgumentTypes(
                I.ParamTypes.size());

            for (auto J = 0; J < I.ParamTypes.size(); J++) {
                if (auto *TensorParamType =
                        std::get_if<TensorParameter>(&I.ParamTypes[J])) {
                    auto Shape =
                        helper.buildShape(TensorParamType->Shape);
                    auto LayoutShape =
                        helper.buildShape(TensorParamType->LayoutShape);

                    auto LayoutFactory = helper.BuildLayoutFactory(
                        LayoutShape, TensorParamType->N_WARPS);
                    auto RegBasis = helper.BuildBasisGroup(
                        LayoutFactory, LayoutFactory.N_REG_AXES,
                        TensorParamType->RegBasis);
                    auto LaneBasis = helper.BuildBasisGroup(
                        LayoutFactory, LayoutFactory.N_LANE_AXES,
                        TensorParamType->LaneBasis);
                    auto WarpBasis = helper.BuildBasisGroup(
                        LayoutFactory, LayoutFactory.N_WARP_AXES,
                        TensorParamType->WarpBasis);

                    auto Layout = helper.BuildLayout(
                        LayoutFactory, RegBasis.first.value(),
                        LaneBasis.first.value(), WarpBasis.first.value());
                    auto TensorType = helper.BuildTensor(
                        getQualTypeFromScalarType(Ctx, TensorParamType->Type),
                        Shape.type, Layout);

                    ArgumentTypes[J] = TensorType.getCanonicalType();
                } else if (auto *ScalarParamType =
                               std::get_if<ScalarType>(&I.ParamTypes[J])) {
                    ArgumentTypes[J] =
                        getQualTypeFromScalarType(Ctx, *ScalarParamType);
                }
            }

            auto FD =
                helper.InstantiateFunction(I.FunctionLookupName, ArgumentTypes);
            if (!FD) {
                llvm::errs() << "ERROR: Failed to instantiate function '"
                             << I.FunctionLookupName << "'\n";
                fail(Ctx);
                return;
            }
            if (FD->isFunctionTemplateSpecialization())
                FD->setTemplateSpecializationKind(
                    clang::TSK_ExplicitInstantiationDefinition);
            CodeGen->HandleTopLevelDecl(clang::DeclGroupRef(FD));
            I.InstFunction = {
                llvm::cast<llvm::Function>(CodeGen->GetAddrOfGlobal(FD, true))};
        }

        CodeGen->HandleTranslationUnit(Ctx);
        ModuleGenCallback(std::move(CodeGen.get()->ReleaseModule()));
    }

private:
    void fail(clang::ASTContext &Ctx) {
        CodeGen->HandleTranslationUnit(Ctx);
        llvm::errs() << "IR generation failed\n";
    }
};

// ============================================================================
// CustomFEAction — FrontendAction creating CustomAstConsumer
// ============================================================================

struct CustomFEAction : clang::ASTFrontendAction {
    llvm::StringRef ModuleName;
    llvm::LLVMContext &Context;
    std::function<void(std::unique_ptr<llvm::Module>)> ModuleGenCallback;
    llvm::SmallVector<DeviceFunctionInstantiation, 4> &InstFunc;

    CustomFEAction(
        const llvm::StringRef &ModuleName, llvm::LLVMContext &Ctx,
        const std::function<void(std::unique_ptr<llvm::Module>)>
            &ModuleGenCallback,
        llvm::SmallVector<DeviceFunctionInstantiation, 4> &InstFunc)
        : ModuleName(ModuleName), Context(Ctx),
          ModuleGenCallback{ModuleGenCallback}, InstFunc{InstFunc} {}

    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) override {
        return std::make_unique<CustomAstConsumer>(Context, ModuleName, CI,
                                                    ModuleGenCallback, InstFunc);
    }
};

// ============================================================================
// CUDACompiler — in-process CUDA compilation using CompilerInvocation
// ============================================================================

struct CUDACompiler {
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> Vfs;
    std::unique_ptr<llvm::WritableMemoryBuffer> SourceBuffer;
    std::unique_ptr<clang::CompilerInstance> CI;

    CUDACompiler(const std::string &SourceCode, int OptLevel,
                 const std::string &sm, const std::string &resourceDir,
                 const std::vector<std::string> &includePaths)
        : Vfs(llvm::vfs::getRealFileSystem()),
          SourceBuffer{std::move(
              llvm::WritableMemoryBuffer::getNewMemBuffer(SourceCode.size()))},
          CI{} {
        std::copy(SourceCode.begin(), SourceCode.end(),
                  SourceBuffer->getBufferStart());

        auto inv = std::make_shared<clang::CompilerInvocation>();

        do {
            auto &L = inv->getLangOpts();
            L.CUDA = true;
            L.CPlusPlus = true;
            L.CPlusPlus11 = true;
            L.CPlusPlus14 = true;
            L.CPlusPlus17 = true;
            L.CPlusPlus20 = true;
            L.Bool = true;
            L.WChar = true;
            L.Char8 = true;
            L.GNUMode = false;
            L.DeclSpecKeyword = true;
            L.GNUAsm = true;
            L.LangStd = clang::LangStandard::lang_cxx20;
            L.CUDAHostDeviceConstexpr = true;
            L.OffloadingNewDriver = true;
            L.DelayedTemplateParsing = true;
            L.Exceptions = false;
            L.CXXExceptions = false;
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
            FrontendOpts.ProgramAction = clang::frontend::ActionKind::ASTDump;
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
                H.AddPath(
                    (resPath / "include").string(),
                    IncludeDirGroup::System, false, true);
                H.AddPath(
                    (resPath / "include" / "cuda_wrapper").string(),
                    IncludeDirGroup::System, false, true);
            }

            H.AddPath("/usr/include/c++/12", IncludeDirGroup::System, false,
                      false);
            H.AddPath("/usr/include/x86_64-linux-gnu/c++/12",
                      IncludeDirGroup::System, false, false);
            H.AddPath("/usr/include/c++/12/backward",
                      IncludeDirGroup::System, false, false);
            H.AddPath("/usr/include", IncludeDirGroup::System, false, true);
            H.AddPath("/usr/include/x86_64-linux-gnu", IncludeDirGroup::System,
                      false, true);

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
};

// ============================================================================
// MLIR spec extraction — scan MLIR module for ttg.extern_call ops
// ============================================================================
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
    llvm::SmallVector<SpecInput, 4> inputs;
};

llvm::SmallVector<ExternCallSpec, 4>
extractExternCallSpecs(mlir::ModuleOp module) {
    using namespace mlir;
    llvm::SmallVector<ExternCallSpec, 4> results;

    StringAttr kRegister = StringAttr::get(module.getContext(), "register");
    StringAttr kLane = StringAttr::get(module.getContext(), "lane");
    StringAttr kWarp = StringAttr::get(module.getContext(), "warp");

    module.walk([&](mlir::triton::gpu::ExternCallOp op) {
        ExternCallSpec spec;
        spec.symbol = op.getSymbol().str();
        spec.libpath = op.getLibpath().str();

        for (auto operand : op.getInputs()) {
            auto tensorTy =
                cast<RankedTensorType>(operand.getType());
            auto shape = tensorTy.getShape();
            auto encoding = tensorTy.getEncoding();

            auto ll = mlir::triton::gpu::toLinearLayout(shape, encoding);

            SpecInput input;
            input.shape.assign(shape.begin(), shape.end());
            input.numWarps = ll.getInDimSize(kWarp);

            auto flattenBases = [](auto bases) {
                llvm::SmallVector<int32_t, 16> flat;
                for (auto &row : bases)
                    flat.append(row.begin(), row.end());
                return flat;
            };

            input.regBases = flattenBases(ll.getBases().lookup(kRegister));
            input.laneBases = flattenBases(ll.getBases().lookup(kLane));
            input.warpBases = flattenBases(ll.getBases().lookup(kWarp));

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

// ============================================================================
// Public API
// ============================================================================

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

std::tuple<std::string, std::string, std::vector<CudaInstantiatedFunc>>
tritonCompileCuda(llvm::LLVMContext &ctx, const std::string &source,
                  const std::string &sm, const std::string &resourceDir,
                  const std::vector<std::string> &includePaths,
                  std::vector<DeviceFunctionInstantiation> &instantiations) {
    std::vector<CudaInstantiatedFunc> results;
    InitializeNVPTXBackend();

    llvm::SmallVector<DeviceFunctionInstantiation, 4> inst(
        instantiations.begin(), instantiations.end());

    CUDACompiler compiler(source, 3, sm, resourceDir, includePaths);

    std::string error;
    std::unique_ptr<llvm::Module> mod;
    {
        std::unique_ptr<llvm::Module> outMod;
        CustomFEAction act(
            "cudamod", ctx,
            [&](std::unique_ptr<llvm::Module> module) {
                outMod = std::move(module);
            },
            inst);

        bool ok = compiler.CI->ExecuteAction(act);
        if (!ok) {
            return {"", "Clang compilation failed", {}};
        }
        if (!outMod) {
            return {"", "Clang compilation produced no module", {}};
        }
        mod = std::move(outMod);
    }

    for (size_t i = 0; i < inst.size(); ++i) {
        auto &I = inst[i];
        if (!I.InstFunction.has_value()) {
            return {"",
                    "Instantiation for '" + I.FunctionLookupName +
                        "' returned null function",
                    {}};
        }
        llvm::Function *fn = I.InstFunction.value();
        CudaInstantiatedFunc r;
        r.Symbol = I.FunctionLookupName;
        r.MangledName = fn->getName().str();
        results.push_back(std::move(r));
    }

    std::string bitcode;
    {
        llvm::raw_string_ostream os(bitcode);
        llvm::WriteBitcodeToFile(*mod, os);
    }

    return {bitcode, "", results};
}

void linkBitcodeToModule(llvm::Module *dstMod, const std::string &bitcode,
                         llvm::LLVMContext &ctx) {
    auto buf = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(bitcode.data(), bitcode.size()));

    // Parse into the same context as dstMod so CloneFunctionInto
    // doesn't leave cross-context metadata references.
    llvm::SMDiagnostic parseErr;
    auto tmpMod = llvm::parseIR(*buf, parseErr, ctx);
    if (!tmpMod) {
        throw std::invalid_argument(
            "Failed to parse CUDA bitcode: " + parseErr.getMessage().str());
    }

    // Collect definitions first (erasing during iteration invalidates
    // the iterator).
    llvm::SmallVector<llvm::Function *, 2> srcDefs;
    for (llvm::Function &F : tmpMod->functions())
        if (!F.isDeclaration())
            srcDefs.push_back(&F);

    for (llvm::Function *srcFn : srcDefs) {
        llvm::Function *dstFn = dstMod->getFunction(srcFn->getName());
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

        // CloneFunctionInto may leave call targets pointing to
        // declarations in the source module (e.g. llvm.lifetime.start).
        // Remap any cross-module callees to their local equivalents
        // so destroying the tmpMod doesn't leave dangling references.
        for (auto &BB : *dstFn) {
            for (auto &I : BB) {
                auto *call = llvm::dyn_cast<llvm::CallInst>(&I);
                if (!call)
                    continue;
                auto *callee = call->getCalledFunction();
                if (!callee || callee->getParent() == dstMod)
                    continue;
                // Look up or create the callee in dstMod
                auto *localFn =
                    dstMod->getFunction(callee->getName());
                if (!localFn) {
                    localFn = llvm::Function::Create(
                        callee->getFunctionType(),
                        callee->getLinkage(),
                        callee->getName(),
                        dstMod);
                }
                call->setCalledFunction(localFn);
            }
        }

        srcFn->eraseFromParent();

        // Fix ret instruction type: CloneFunctionInto keeps the named
        // struct (%struct.Tensor) from the source module, but the
        // function's declared return type uses the literal struct
        // ({ [16 x float] }) from MLIR.  Extract each field and insert
        // into an undef literal struct to bridge the mismatch.
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
                auto *dstST = llvm::cast<llvm::StructType>(funcRetTy);
                llvm::Value *newVal =
                    llvm::UndefValue::get(funcRetTy);
                for (unsigned i = 0; i < srcST->getNumElements(); ++i) {
                    auto *field = llvm::ExtractValueInst::Create(
                        retVal, {i}, "", ret->getIterator());
                    // If field types differ (same LLVMContext but
                    // different Type* created by clone vs MLIR),
                    // use an alloca to "launder" the type.
                    // Opaque pointer LLVM allows store/load
                    // of differently-named but structurally
                    // identical types.
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
                        newVal, castField, {i}, "",
                        ret->getIterator());
                }
                ret->setOperand(0, newVal);
            }
        }

        // Make the DISubprogram distinct if needed.
        if (auto *sub = dstFn->getSubprogram()) {
            if (!sub->isDistinct()) {
                auto &llCtx = dstFn->getContext();
                auto *newSub = llvm::DISubprogram::getDistinct(
                    llCtx, sub->getScope(), sub->getName(),
                    sub->getLinkageName(), sub->getFile(),
                    sub->getLine(), sub->getType(),
                    sub->getScopeLine(), sub->getContainingType(),
                    sub->getVirtualIndex(), sub->getThisAdjustment(),
                    sub->getFlags(), sub->getSPFlags(), sub->getUnit());
                dstFn->setSubprogram(newSub);
            }
        }

        dstFn->addFnAttr(llvm::Attribute::AlwaysInline);
    }

    llvm::StripDebugInfo(*dstMod);
}
