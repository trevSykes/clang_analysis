#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "tensorCompAnalysis.h" // Add this line

using namespace clang::tooling;
using namespace llvm;

// Apply a custom category to all command-line options
static cl::OptionCategory TCACategory("Tensor Computation Analysis options");

int main(int argc, const char **argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, TCACategory);
    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }
    CommonOptionsParser& OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(),
                  OptionsParser.getSourcePathList());

    return Tool.run(newFrontendActionFactory<TensorOutputAction>().get());
}