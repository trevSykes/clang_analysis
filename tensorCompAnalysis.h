#ifndef TENSOR_COMP_ANALYSIS_H
#define TENSOR_COMP_ANALYSIS_H

#include "clang/Frontend/FrontendAction.h"

class TensorOutputAction : public clang::PluginASTAction {
protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &CI, llvm::StringRef) override;
    
    bool ParseArgs(const clang::CompilerInstance &CI,
                  const std::vector<std::string>& args) override;
};

#endif