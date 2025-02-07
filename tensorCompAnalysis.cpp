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
#include <set>
#include <algorithm>
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

// Structure to record for–loop information.
struct LoopInfo {
    std::string varName;     // Name of the loop variable.
    unsigned loopDepth;      // The depth at which the loop was encountered.
};

bool loopIsAnInitialiser(ForStmt *FS) {
    // Print for debugging.
    // llvm::errs() << "[DEBUG] Checking if loop is an initialiser\n";
    Stmt *Body = FS->getBody();
    if (!Body)
        return false;
    if (auto *CS = dyn_cast<CompoundStmt>(Body)) {
        for (auto *S : CS->body()) {
            if (auto *BO = dyn_cast<BinaryOperator>(S)) {
                if (BO->isAssignmentOp()) {
                    Expr *RHS = BO->getRHS();
                    if (auto *IL = dyn_cast<IntegerLiteral>(RHS)) {
                        // llvm::errs() << "[DEBUG] Found an integer literal\n";
                    } else if (auto *FL = dyn_cast<FloatingLiteral>(RHS)){
                        // llvm::errs() << "[DEBUG] Found a floating literal\n";
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    }
    return false;
}

class TensorOutputVisitor : public RecursiveASTVisitor<TensorOutputVisitor> {
public:
    explicit TensorOutputVisitor(ASTContext *Context)
      : Context(Context), currentLoopDepth(0) {}

    // ------------------------------------------------------------------------
    // 1. VisitUnaryOperator: record pointer update (e.g. for ++/--)
    // Instead of using the global currentLoopDepth we record the entire active loop stack.
    bool VisitUnaryOperator(UnaryOperator *UO) {
        if (UO->isIncrementDecrementOp()) { // catches ++/--
            if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(UO->getSubExpr()->IgnoreParenImpCasts())) {
                std::string varName = DRE->getNameInfo().getAsString();
                // Increase our candidate counter.
                candidateAssignments[varName]++;
                // Instead of recording only currentLoopDepth, record all active loops.
                for (const auto &li : loopStack) {
                    pointerUpdateActiveLoops[varName].insert({li.loopDepth, li.varName});
                }
            }
        }
        return true;
    }
    
    // ------------------------------------------------------------------------
    // Process function declarations.
    bool VisitFunctionDecl(FunctionDecl *FD) {

        // Only process functions with a body.
        if (!FD->hasBody())
            return true;

        // Reset state at the beginning of each function.
        candidateAssignments.clear();
        loopStack.clear();
        allLoops.clear();
        initRecords.clear();
        assignmentRecords.clear();
        currentLoopDepth = 0;
        pointerUpdates.clear();
        pointerResets.clear();
        pointerUpdateActiveLoops.clear();

        // Record parameters.
        for (unsigned i = 0; i < FD->getNumParams(); i++) {
            VisitParmVarDecl(FD->getParamDecl(i));
        }
        // Traverse the body.
        TraverseStmt(FD->getBody());

        // Choose the candidate output variable via our heuristic.
        std::string candidateOutput = chooseOutputVariable();

        // Build JSON output.
        Json::Value result;
        result["candidate_output"] = candidateOutput;

        // --- Output the recorded initializations.
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

        // --- Output the assignment records.
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

        // --- Now, infer and report the output tensor's dimensionality.
        // Determine if the candidate is used as a pointer or as an array.
        bool isArrayCandidate = false;
        for (const auto &rec : assignmentRecords) {
            // If any assignment record for the candidate contains an index expression,
            // assume that the candidate is used as an array.
            if (rec.varName == candidateOutput && !rec.indexExpr.empty()) {
                isArrayCandidate = true;
                break;
            }
        }
        
        Json::Value dims(Json::arrayValue);
        if (!isArrayCandidate) {
            // --- (A) Candidate is a pointer.
            // Use the union of active loop contexts recorded at pointer update.
            std::vector<std::pair<unsigned, std::string>> dimsVec;
            if (pointerUpdateActiveLoops.find(candidateOutput) != pointerUpdateActiveLoops.end()) {
                for (auto &entry : pointerUpdateActiveLoops[candidateOutput]) {
                    dimsVec.push_back(entry);
                }
            }
            // Sort by loop depth.
            std::sort(dimsVec.begin(), dimsVec.end(), [](const auto &a, const auto &b) {
                return a.first < b.first;
            });
            // Report each active loop as a tensor dimension.
            for (auto &p : dimsVec) {
                Json::Value dim;
                dim["loopDepth"] = static_cast<int>(p.first);
                dim["loopVar"] = p.second;
                dims.append(dim);
            }
        } else {
            // --- (B) Candidate is an array.
            // Collapse the index expressions from assignment records.
            std::string combinedIndexExpr;
            for (const auto &rec : assignmentRecords) {
                if (rec.varName == candidateOutput && !rec.indexExpr.empty()) {
                    combinedIndexExpr += rec.indexExpr + " ";
                }
            }
            // Helper lambda: substitute local variable names with their initializer text.
            auto collapseIndexExpr = [&](const std::string &expr) -> std::string {
                std::string collapsed = expr;
                for (const auto &rec : initRecords) {
                    if (rec.initType == "local" && !rec.initValue.empty()) {
                        size_t pos = 0;
                        while ((pos = collapsed.find(rec.varName, pos)) != std::string::npos) {
                            collapsed.replace(pos, rec.varName.size(), rec.initValue);
                            pos += rec.initValue.size();
                        }
                    }
                }
                return collapsed;
            };
            // Helper lambda: extract loop variable names from the collapsed expression.
            auto extractLoopVars = [&](const std::string &expr) -> std::vector<std::string> {
                std::vector<std::string> loopVars;
                for (const auto &li : allLoops) {
                    if (expr.find(li.varName) != std::string::npos)
                        loopVars.push_back(li.varName);
                }
                return loopVars;
            };

            std::string collapsedExpr = collapseIndexExpr(combinedIndexExpr);
            auto loopVars = extractLoopVars(collapsedExpr);
            std::set<std::string> uniqueLoopVars(loopVars.begin(), loopVars.end());
            for (const auto &lv : uniqueLoopVars) {
                Json::Value dim;
                dim["loopVar"] = lv;
                for (const auto &li : allLoops) {
                    if (li.varName == lv) {
                        dim["loopDepth"] = static_cast<int>(li.loopDepth);
                        break;
                    }
                }
                dims.append(dim);
            }
        }
        result["tensor_dimensions"] = dims;

        llvm::outs() << result.toStyledString() << "\n";
        return true;
    }


    // ------------------------------------------------------------------------
    // Traverse for-loop statements.
    // Record every loop in the vector 'allLoops' (as well as push/pop the temporary loopStack).
    bool TraverseForStmt(ForStmt *FS) {
        if (loopIsAnInitialiser(FS)) {
            //Declare which loop we are skipping
            // llvm::errs() << "[DEBUG] Skipping loop";
            // FS->getInit()->dump();
            return RecursiveASTVisitor<TensorOutputVisitor>::TraverseForStmt(FS);
        }
        // llvm::errs() << "[DEBUG] YOOOHOOO\n";
        // llvm::errs() << "[DEBUG] Visiting loop with index" << currentLoopDepth << "\n";
        if (DeclStmt *DS = dyn_cast<DeclStmt>(FS->getInit())) {
            for (auto it = DS->decl_begin(); it != DS->decl_end(); ++it) {
                if (VarDecl *VD = dyn_cast<VarDecl>(*it)) {
                    LoopInfo li;
                    li.varName = VD->getNameAsString();
                    li.loopDepth = currentLoopDepth;
                    loopStack.push_back(li);
                    allLoops.push_back(li);
                    currentLoopDepth++;
                    
                    // Record an initialization for this loop variable.
                    InitRecord rec;
                    rec.varName = li.varName;
                    rec.initType = "for_loop";
                    rec.loopDepth = li.loopDepth;
                    rec.initValue = "";
                    rec.lineNumber = Context->getSourceManager().getSpellingLineNumber(VD->getBeginLoc());
                    initRecords.push_back(rec);

                    TraverseStmt(FS->getBody());
                    currentLoopDepth--;
                    loopStack.pop_back();
                    return true;
                }
            }
        } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(FS->getInit())) {
            if (BO->isAssignmentOp()) {
                if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts())) {
                    LoopInfo li;
                    li.varName = DRE->getNameInfo().getAsString();
                    li.loopDepth = currentLoopDepth;
                    loopStack.push_back(li);
                    allLoops.push_back(li);
                    currentLoopDepth++;

                    InitRecord rec;
                    rec.varName = li.varName;
                    rec.initType = "for_loop";
                    rec.loopDepth = li.loopDepth;
                    rec.initValue = "";
                    rec.lineNumber = Context->getSourceManager().getSpellingLineNumber(DRE->getBeginLoc());
                    initRecords.push_back(rec);

                    TraverseStmt(FS->getBody());
                    currentLoopDepth--;
                    loopStack.pop_back();
                    return true;
                }
            }
        }
        return RecursiveASTVisitor<TensorOutputVisitor>::TraverseForStmt(FS);
    }

    // ------------------------------------------------------------------------
    // Record local variable declarations.
    bool VisitVarDecl(VarDecl *VD) {
        if (!VD->isLocalVarDecl())
            return true;
        InitRecord rec;
        rec.varName = VD->getNameAsString();
        rec.initType = "local";
        rec.loopDepth = currentLoopDepth;
        rec.initValue = "";
        if (VD->hasInit()) {
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

    // ------------------------------------------------------------------------
    // Record parameter declarations.
    bool VisitParmVarDecl(ParmVarDecl *PVD) {    
        std::string paramName = PVD->getNameAsString();
        for (const auto &rec : initRecords) {
            if (rec.varName == paramName)
                return true; // Avoid duplicate entries.
        }
        InitRecord rec;
        rec.varName = paramName;
        rec.initType = "parameter";
        rec.loopDepth = currentLoopDepth;
        rec.initValue = "";
        rec.lineNumber = Context->getSourceManager().getSpellingLineNumber(PVD->getBeginLoc());
        initRecords.push_back(rec);
        return true;
    }

    // ------------------------------------------------------------------------
    // Record assignments and detect pointer resets.
    bool VisitBinaryOperator(BinaryOperator *BO) {
    if (BO->isAssignmentOp() || BO->isCompoundAssignmentOp()) {
        AssignmentRecord rec;
        rec.loopDepth = currentLoopDepth;
        rec.lineNumber = Context->getSourceManager().getSpellingLineNumber(BO->getOperatorLoc());
        rec.arrayName = "";
        rec.indexExpr = "";

        // Check for array subscript on the LHS.
        if (ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(BO->getLHS())) {
            if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreParenImpCasts())) {
                rec.varName = DRE->getNameInfo().getAsString();
                rec.arrayName = rec.varName;
                auto &SM = Context->getSourceManager();
                SourceLocation Start = ASE->getIdx()->getBeginLoc();
                SourceLocation End = ASE->getIdx()->getEndLoc();
                CharSourceRange range = CharSourceRange::getTokenRange(Start, End);
                rec.indexExpr = Lexer::getSourceText(range, SM, LangOptions()).str();
            }
        }
        // If LHS is a direct DeclRefExpr.
        else if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts())) {
            rec.varName = DRE->getNameInfo().getAsString();
        }
        // NEW: If LHS is a UnaryOperator (e.g., a dereference "*p_c")
        else if (UnaryOperator *UO = dyn_cast<UnaryOperator>(BO->getLHS()->IgnoreParenImpCasts())) {
            if (UO->getOpcode() == UO_Deref) {
                if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(UO->getSubExpr()->IgnoreParenImpCasts())) {
                    rec.varName = DRE->getNameInfo().getAsString();
                }
            }
        }

        // Get RHS expression as source text.
        if (BO->getRHS()) {
            auto &SM = Context->getSourceManager();
            SourceLocation Start = BO->getRHS()->getBeginLoc();
            SourceLocation End = BO->getRHS()->getEndLoc();
            CharSourceRange range = CharSourceRange::getTokenRange(Start, End);
            rec.valueExpr = Lexer::getSourceText(range, SM, LangOptions()).str();
            // If RHS is an address-of operator, count as a pointer reset.
            if (UnaryOperator *UO = dyn_cast<UnaryOperator>(BO->getRHS()->IgnoreParenImpCasts())) {
                if (UO->getOpcode() == UO_AddrOf)
                    pointerResets[rec.varName]++;
            }
        } else {
            rec.valueExpr = "";
        }
        if (!rec.varName.empty())
            candidateAssignments[rec.varName]++;
        assignmentRecords.push_back(rec);
    }
    return true;
}


private:
    ASTContext *Context;
    unsigned currentLoopDepth;
    std::vector<LoopInfo> loopStack;  // active loop stack (for current context)
    std::vector<LoopInfo> allLoops;   // all encountered loops

    // Data for initializations and assignments.
    std::vector<InitRecord> initRecords;
    std::vector<AssignmentRecord> assignmentRecords;
    
    // For pointer analysis.
    // (pointerUpdates is kept for legacy purposes, though we now use pointerUpdateActiveLoops)
    std::map<std::string, std::vector<unsigned>> pointerUpdates;
    std::map<std::string, unsigned> pointerResets; // count pointer resets
    // New: record the full set of active loops at each pointer update.
    std::map<std::string, std::set<std::pair<unsigned, std::string>>> pointerUpdateActiveLoops;

    // For candidate selection.
    std::map<std::string, unsigned> candidateAssignments;

    // Choose candidate output variable based on assignment counts (and ignore those that reset).
    std::string chooseOutputVariable() {
        std::string candidate;
        unsigned maxScore = 0;
        for (auto &entry : candidateAssignments) {
            unsigned resets = pointerResets[entry.first];
            if (resets > 0)
                continue;
            unsigned score = entry.second;
            if (pointerUpdates.count(entry.first))
                score += pointerUpdates[entry.first].size();
            if (score > maxScore) {
                maxScore = score;
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
