#pragma once
#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/framework/ex_lib_loader.h"

namespace onnxruntime {

class FuncManager {
 public:
  FuncManager() : fused_funcs_(std::make_shared<std::unordered_map<std::string, FuncInfo> >()), lib_loader_(onnxruntime::make_unique<ExLibLoader>()) {}

  Status AddFuncInfo(const std::string& name, const std::string& dll_path);

  Status AddFuncInfo(const std::string& name, ComputeFunc compute, CreateFunctionStateFunc create, DestroyFunctionStateFunc release);

  Status GetFuncs(const std::string& name, ComputeFunc* compute, CreateFunctionStateFunc* create, DestroyFunctionStateFunc* release) const;

  void SetFusedFuncs(const FuncManager& func_mgr) {
    fused_funcs_ = func_mgr.fused_funcs_;
  }

  struct FuncInfo {
    std::string dso_path;
    ComputeFunc compute_func;
    CreateFunctionStateFunc create_state_func;
    DestroyFunctionStateFunc release_state_func;
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
