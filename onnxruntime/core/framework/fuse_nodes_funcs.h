#pragma once
#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/framework/ex_lib_loader.h"

namespace onnxruntime {

class FuseFuncManager {
 public:
  FuseFuncManager() : lib_loader_(std::make_unique<ExLibLoader>()), fused_funcs_(std::make_unique<std::unordered_map<std::string, FuncInfo> >()){}

  Status AddFuncInfo(const std::string& name, const std::string& dll_path);

  Status AddFuncInfo(const std::string& name, ComputeFunc compute, CreateFunctionState create, ReleaseFunctionState release);

  Status GetFuncs(const std::string& name, ComputeFunc* compute, CreateFunctionState* create, ReleaseFunctionState* release) const;

  struct FuncInfo {
    std::string dso_path;
    ComputeFunc compute_func;
    CreateFunctionState create_state_func;
    ReleaseFunctionState release_state_func;
  };

 private:
  const std::string kComputeFuncSymbol = "Compute_";
  const std::string kCreateStateFuncSymbol = "Create_State_";
  const std::string kReleaseStateFuncSymbol = "Release_State_";

  std::unique_ptr<std::unordered_map<std::string, FuncInfo> > fused_funcs_;
  std::unique_ptr<ExLibLoader> lib_loader_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FuseFuncManager);
};
}  // namespace onnxruntime
