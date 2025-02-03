#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>
#include <vector>
#include <json/json.h> 

using namespace clang;

// Structure to record variable initializations.
struct InitRecord {
    std::string varName;
    std::string initType;    // "local", "parameter", or "for_loop"
    std::string initValue;   // The initializer expression as source text (if any)
    unsigned loopDepth;      // 0 = top-level; 1 = inside first loop; etc.
    unsigned lineNumber;     // Source line number of initialization
};

// Structure to record assignment information.
struct AssignmentRecord {
    std::string varName;     // Name of the variable being assigned to.
    std::string arrayName;   // If the LHS is an index expression, record the array name.
    std::string indexExpr;   // If the LHS is an index expression, record the index expression text.
    std::string valueExpr;   // The RHS expression as source text.
    unsigned loopDepth;      // Loop depth at assignment.
    unsigned lineNumber;     // Source line number of the assignment.
};

struct LoopInfo {
    std::string varName;
    unsigned loopDepth;
};

class TensorOutputVisitor : public RecursiveASTVisitor<TensorOutputVisitor> {
public:
    explicit TensorOutputVisitor(ASTContext *Context)
      : Context(Context), currentLoopDepth(0) {}

    bool VisitFunctionDecl(FunctionDecl *FD) {
        // Only process functions with a body.
        if (!FD->hasBody())
            return true;

        // Reset state at the beginning of each function.
        candidateAssignments.clear();
        loopStack.clear();
        initRecords.clear();
        assignmentRecords.clear();
        currentLoopDepth = 0;

        for (unsigned i = 0; i < FD->getNumParams(); i++) {
            VisitParmVarDecl(FD->getParamDecl(i));
        }
            TraverseStmt(FD->getBody());

        // Choose the candidate output variable via maximum assignments.
        std::string candidateOutput = chooseOutputVariable();

        // Build JSON output.
        Json::Value result;
        result["candidate_output"] = candidateOutput;

        // Build initialization map JSON.
        Json::Value initMapJson(Json::arrayValue);
        for (const auto &initRec : initRecords) {
            Json::Value recJson;
            recJson["varName"] = initRec.varName;
            recJson["initType"] = initRec.initType;
            recJson["initValue"] = initRec.initValue;
            recJson["loopDepth"] = static_cast<int>(initRec.loopDepth);
            recJson["lineNumber"] = static_cast<int>(initRec.lineNumber);
            initMapJson.append(recJson);
        }
        result["init_map"] = initMapJson;

        // Build assignment map JSON.
        Json::Value assignMapJson(Json::arrayValue);
        for (const auto &assignRec : assignmentRecords) {
            Json::Value recJson;
            recJson["varName"] = assignRec.varName;
            recJson["arrayName"] = assignRec.arrayName;
            recJson["indexExpr"] = assignRec.indexExpr;
            recJson["valueExpr"] = assignRec.valueExpr;
            recJson["loopDepth"] = static_cast<int>(assignRec.loopDepth);
            recJson["lineNumber"] = static_cast<int>(assignRec.lineNumber);
            assignMapJson.append(recJson);
        }
        result["assignment_map"] = assignMapJson;

        llvm::outs() << result.toStyledString() << "\n";
        return true;
    }

    // Traverse for-loop statements to record loop variable initializations.
    bool TraverseForStmt(ForStmt *FS) {
        // Try to extract the loop variable from the for-loop initializer.
        if (DeclStmt *DS = dyn_cast<DeclStmt>(FS->getInit())) {
            for (auto it = DS->decl_begin(); it != DS->decl_end(); ++it) {
                if (VarDecl *VD = dyn_cast<VarDecl>(*it)) {
                    // Record loop variable information.
                    LoopInfo li;
                    li.varName = VD->getNameAsString();
                    li.loopDepth = currentLoopDepth;
                    loopStack.push_back(li);
                    currentLoopDepth++;

                    // Record an initialization for this loop variable.
                    llvm::errs() << "[DEBUG] Found loop variable: " << VD->getNameAsString() << "\n";
                    InitRecord rec;
                    rec.varName = VD->getNameAsString();
                    rec.initType = "for_loop";
                    rec.loopDepth = currentLoopDepth - 1;
                    rec.initValue = "";
                    // Get the line number.
                    rec.lineNumber = Context->getSourceManager().getSpellingLineNumber(VD->getBeginLoc());
                    initRecords.push_back(rec);

                    // Traverse the loop body.
                    TraverseStmt(FS->getBody());
                    currentLoopDepth--;
                    loopStack.pop_back();
                    return true;
                }
            }
        } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(FS->getInit())) {
            // Handle the case where the loop variable is initialized in the condition.
            if (BO->isAssignmentOp()) {
                if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts())) {
                    // Record loop variable information.
                    LoopInfo li;
                    li.varName = DRE->getNameInfo().getAsString();
                    li.loopDepth = currentLoopDepth;
                    loopStack.push_back(li);
                    currentLoopDepth++;

                    // Record an initialization for this loop variable.
                    llvm::errs() << "[DEBUG] Found loop variable: " << DRE->getNameInfo().getAsString() << "\n";
                    InitRecord rec;
                    rec.varName = DRE->getNameInfo().getAsString();
                    rec.initType = "for_loop";
                    rec.loopDepth = currentLoopDepth - 1;
                    rec.initValue = "";
                    // Get the line number.
                    rec.lineNumber = Context->getSourceManager().getSpellingLineNumber(DRE->getBeginLoc());
                    initRecords.push_back(rec);

                    // Traverse the loop body.
                    TraverseStmt(FS->getBody());
                    currentLoopDepth--;
                    loopStack.pop_back();
                    return true;
                }
            }
            llvm::errs() << "[DEBUG] For loop initializer is not a DeclStmt, it is: " 
                         << FS->getInit()->getStmtClassName() << "\n";
        }
        // Fallback: if we cannot extract a loop variable, just traverse the children.
        return RecursiveASTVisitor<TensorOutputVisitor>::TraverseForStmt(FS);
    }

    // Record local variable declarations.
    bool VisitVarDecl(VarDecl *VD) {
        // Only record declarations within function bodies.
        if (!VD->isLocalVarDecl())
            return true;

        InitRecord rec;
        rec.varName = VD->getNameAsString();
        rec.initType = "local";
        rec.loopDepth = currentLoopDepth;
        rec.initValue = "";
        if (VD->hasInit()) {
            // Get the initializer text.
            auto &SM = Context->getSourceManager();
            SourceLocation Start = VD->getInit()->getBeginLoc();
            SourceLocation End = VD->getInit()->getEndLoc();
            CharSourceRange range = CharSourceRange::getTokenRange(Start, End);
            rec.initValue = Lexer::getSourceText(range, SM, LangOptions()).str();
        }
        rec.lineNumber = Context->getSourceManager().getSpellingLineNumber(VD->getBeginLoc());
        initRecords.push_back(rec);
        return true;
    }

    // Record parameter declarations.
    bool VisitParmVarDecl(ParmVarDecl *PVD) {    
        std::string paramName = PVD->getNameAsString();

        // Check if the parameter already exists in initRecords
        for (const auto &rec : initRecords) {
             if (rec.varName == paramName) {
            return true; // Avoid duplicate entries
               }
              }       
        InitRecord rec;
        rec.varName = PVD->getNameAsString();
        rec.initType = "parameter";
        rec.loopDepth = currentLoopDepth; // likely 0 for parameters
        rec.initValue = "";  // parameters do not have an initializer in the function body.
        rec.lineNumber = Context->getSourceManager().getSpellingLineNumber(PVD->getBeginLoc());
        initRecords.push_back(rec);
        return true;
    }

    // Record assignment and compound assignment operations.
    bool VisitBinaryOperator(BinaryOperator *BO) {
        if (BO->isAssignmentOp() || BO->isCompoundAssignmentOp()) {
            AssignmentRecord rec;
            rec.loopDepth = currentLoopDepth;
            rec.lineNumber = Context->getSourceManager().getSpellingLineNumber(BO->getOperatorLoc());

            // Default: no array indexing.
            rec.arrayName = "";
            rec.indexExpr = "";

            // Check if the LHS is an array subscript.
            if (ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(BO->getLHS())) {
                // For array assignments, record the array base variable.
                if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreParenImpCasts())) {
                    rec.varName = DRE->getNameInfo().getAsString();
                    rec.arrayName = rec.varName;
                    // Get the index expression as source text.
                    auto &SM = Context->getSourceManager();
                    SourceLocation Start = ASE->getIdx()->getBeginLoc();
                    SourceLocation End = ASE->getIdx()->getEndLoc();
                    CharSourceRange range = CharSourceRange::getTokenRange(Start, End);
                    rec.indexExpr = Lexer::getSourceText(range, SM, LangOptions()).str();
                }
            }
            // Otherwise, if the LHS is a plain reference.
            else if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts())) {
                rec.varName = DRE->getNameInfo().getAsString();
            }

            // Get the RHS expression as source text.
            if (BO->getRHS()) {
                auto &SM = Context->getSourceManager();
                SourceLocation Start = BO->getRHS()->getBeginLoc();
                SourceLocation End = BO->getRHS()->getEndLoc();
                CharSourceRange range = CharSourceRange::getTokenRange(Start, End);
                rec.valueExpr = Lexer::getSourceText(range, SM, LangOptions()).str();
            } else {
                rec.valueExpr = "";
            }

            // Increment our assignment counter for candidate selection.
            if (!rec.varName.empty()) {
                candidateAssignments[rec.varName]++;
            }
            assignmentRecords.push_back(rec);
        }
        return true;
    }

private:
    ASTContext *Context;
    unsigned currentLoopDepth;
    std::vector<LoopInfo> loopStack;
    // Vectors to hold our maps.
    std::vector<InitRecord> initRecords;
    std::vector<AssignmentRecord> assignmentRecords;

    // Map to count assignments per variable (for candidate selection).
    std::map<std::string, unsigned> candidateAssignments;

    // Choose the candidate output variable based on the maximum number of assignments.
    std::string chooseOutputVariable() {
        std::string candidate;
        unsigned maxAssign = 0;
        for (auto &entry : candidateAssignments) {
            if (entry.second > maxAssign) {
                maxAssign = entry.second;
                candidate = entry.first;
            }
        }
        return candidate;
    }
};

class TensorOutputConsumer : public clang::ASTConsumer {
public:
    explicit TensorOutputConsumer(ASTContext *Context)
        : Visitor(Context) {}

    virtual void HandleTranslationUnit(clang::ASTContext &Context) {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
private:
    TensorOutputVisitor Visitor;
};

class TensorOutputAction : public clang::PluginASTAction {
protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &CI, llvm::StringRef) override {
        return std::make_unique<TensorOutputConsumer>(&CI.getASTContext());
    }

    bool ParseArgs(const clang::CompilerInstance &CI,
                   const std::vector<std::string>& args) override {
        return true;
    }
};

// Registration of the plugin.
static clang::FrontendPluginRegistry::Add<TensorOutputAction>
X("TCA", "Tensor Computation Analysis Plugin");
