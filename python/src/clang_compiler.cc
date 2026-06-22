#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Host.h"
#include <cstdio>
#include <memory>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

std::unique_ptr<llvm::Module>
compileCudaToModule(const std::string &source, const std::string &filename,
                    const std::vector<std::string> &args,
                    llvm::LLVMContext &tritonCtx, std::string &error) {
  // Write source to a real temp file (VFS overlay has issues with the Driver).
  std::string tmpname = "/tmp/triton_clang_" + filename + ".XXXXXX.cu";
  std::vector<char> tmpbuf(tmpname.data(), tmpname.data() + tmpname.size() + 1);
  int fd = mkstemps(tmpbuf.data(), 3); // .cu = 3 chars
  if (fd < 0) {
    error = "Failed to create temp file";
    return nullptr;
  }
  tmpname.assign(tmpbuf.data());
  FILE *fp = fdopen(fd, "w");
  if (!fp) {
    error = "Failed to open temp file";
    close(fd);
    return nullptr;
  }
  fwrite(source.data(), 1, source.size(), fp);
  fclose(fp);

  std::string diagBuf;
  llvm::raw_string_ostream diagOS(diagBuf);

  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagIDs(
      new clang::DiagnosticIDs());
  clang::DiagnosticOptions diagOpts;
  clang::TextDiagnosticPrinter printer(diagOS, diagOpts);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags(
      new clang::DiagnosticsEngine(diagIDs, diagOpts, &printer,
                                   /*ShouldOwnClient=*/false));

  std::vector<const char *> argv;
  argv.push_back("clang");
  for (const auto &a : args)
    argv.push_back(a.c_str());
  argv.push_back(tmpname.c_str());

  // Use the driver to build the compilation
  clang::driver::Driver driver(
      "clang", "nvptx64-nvidia-cuda",
      *diags, "triton-clang");
  driver.setTitle("triton-clang");

  std::unique_ptr<clang::driver::Compilation> compilation(
      driver.BuildCompilation(argv));
  diagOS.flush();

  if (!compilation) {
    error = "Driver failed to build compilation";
    if (!diagBuf.empty())
      error += ": " + diagBuf;
    unlink(tmpname.c_str());
    return nullptr;
  }

  const llvm::opt::ArgStringList *cc1Args = nullptr;
  const clang::driver::JobList &jobs = compilation->getJobs();
  for (const auto &job : jobs) {
    if (llvm::StringRef(job.getCreator().getName()) == "clang") {
      cc1Args = &job.getArguments();
      break;
    }
  }
  if (!cc1Args) {
    error = "No clang CC1 job found in compilation";
    if (!diagBuf.empty())
      error += ": " + diagBuf;
    unlink(tmpname.c_str());
    return nullptr;
  }

  auto invocation = std::make_shared<clang::CompilerInvocation>();
  if (!clang::CompilerInvocation::CreateFromArgs(*invocation, *cc1Args,
                                                  *diags, "clang")) {
    error = "Failed to create CompilerInvocation";
    if (!diagBuf.empty())
      error += ": " + diagBuf;
    unlink(tmpname.c_str());
    return nullptr;
  }

  auto ci = std::make_unique<clang::CompilerInstance>(
      invocation, std::make_shared<clang::PCHContainerOperations>());
  ci->setDiagnostics(diags.get());

  auto action = std::make_unique<clang::EmitLLVMOnlyAction>(&tritonCtx);
  bool ok = ci->ExecuteAction(*action);
  if (!ok) {
    error = "Clang compilation failed";
    if (!diagBuf.empty())
      error += ": " + diagBuf;
    unlink(tmpname.c_str());
    return nullptr;
  }

  unlink(tmpname.c_str());

  auto mod = action->takeModule();
  if (!mod) {
    error = "Clang produced no module";
    return nullptr;
  }

  // Explicitly destroy clang objects before returning Module.
  // This ensures Clang doesn't hold references to LLVMContext data
  // that the Module depends on.
  action.reset();
  ci.reset();
  invocation.reset();
  compilation.reset();

  return mod;
}

} // namespace

std::pair<std::unique_ptr<llvm::Module>, std::string>
tritonCompileCudaToModule(llvm::LLVMContext &ctx, const std::string &source,
                          const std::string &filename,
                          const std::vector<std::string> &args) {
  std::string error;
  auto mod = compileCudaToModule(source, filename, args, ctx, error);
  return {std::move(mod), error};
}
