#pragma once
#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/framework/ex_lib_loader.h"

namespace onnxruntime {

class FuncManager {
 public:
  FuncManager()
      : fused_funcs_(std::make_shared<std::unordered_map<std::string, FuncInfo> >()),
        lib_loader_(std::make_unique<ExLibLoader>()) {
  }

  Status AddFuncInfo(const std::string& name, const std::string& dll_path);

  Status AddFuncInfo(const std::string& name, NodeComputeInfo&& compute_info);

  Status GetFuncs(const std::string& name, NodeComputeInfo*& compute_info) const;

  size_t NumFuncs() const { return fused_funcs_->size(); }

  void SetFusedFuncs(const FuncManager& func_mgr) {
    fused_funcs_ = func_mgr.fused_funcs_;
  }

  struct FuncInfo {
    std::string dso_path;
    NodeComputeInfo compute_info;
  };

 private:
  const std::string kComputeFuncSymbol = "Compute_";
  const std::string kCreateStateFuncSymbol = "Create_State_";
  const std::string kReleaseStateFuncSymbol = "Release_State_";

  // note that subgraph session state shares fused_funcs with main graph
  // because it's filled in by the time main graph is traversed,
  // while subgraph session state is created later
  std::shared_ptr<std::unordered_map<std::string, FuncInfo> > fused_funcs_;
  std::unique_ptr<ExLibLoader> lib_loader_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FuncManager);
};
}  // namespace onnxruntime
