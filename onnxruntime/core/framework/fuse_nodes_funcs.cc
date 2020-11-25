#include "core/framework/fuse_nodes_funcs.h"
#include "core/platform/env.h"

namespace onnxruntime {
Status FuncManager::AddFuncInfo(const std::string& name, const std::string& dll_path) {
  auto it = fused_funcs_->find(name);
  if (it != fused_funcs_->end())
    return Status(common::ONNXRUNTIME, common::FAIL, "func info for node: " + name + " already exist.");
  (*fused_funcs_)[name] = {dll_path, NodeComputeInfo()};
  return Status::OK();
}

Status FuncManager::AddFuncInfo(const std::string& name, NodeComputeInfo&& compute_info) {
  auto it = fused_funcs_->find(name);
  if (it != fused_funcs_->end())
    return Status(common::ONNXRUNTIME, common::FAIL, "func info for node: " + name + " already exist.");

  if (!compute_info.compute_func || !compute_info.create_state_func || !compute_info.release_state_func)
    return Status(common::ONNXRUNTIME, common::FAIL, "Can't use func with null ptr");

  (*fused_funcs_)[name] = {"", std::move(compute_info)};
  return Status::OK();
}

Status FuncManager::GetFuncs(const std::string& name, NodeComputeInfo*& compute_info) const {
  auto it = fused_funcs_->find(name);
  if (it == fused_funcs_->end())
    return Status(common::ONNXRUNTIME, common::FAIL, "func info for node: " + name + " not found.");

  if (!it->second.compute_info.compute_func) {
    //load from path
    void* handle = nullptr;
    ORT_RETURN_IF_ERROR(lib_loader_->LoadExternalLib(it->second.dso_path, &handle));
    void* create_func_symbol_handle = nullptr;
    ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle,
                                                            kCreateStateFuncSymbol + name,
                                                            &create_func_symbol_handle));
    void* compute_func_symbol_handle = nullptr;
    ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle,
                                                            kComputeFuncSymbol + name,
                                                            &compute_func_symbol_handle));
    void* release_func_symbol_handle = nullptr;
    ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle,
                                                            kReleaseStateFuncSymbol + name,
                                                            &release_func_symbol_handle));
    it->second.compute_info.compute_func = [=](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      return reinterpret_cast<ComputeFuncC>(compute_func_symbol_handle)(state, api, context);
    };

    it->second.compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      return reinterpret_cast<CreateFunctionStateC>(create_func_symbol_handle)(context, state);
    };

    it->second.compute_info.release_state_func = [=](FunctionState state) {
      return reinterpret_cast<DestroyFunctionStateC>(release_func_symbol_handle)(state);
    };
  }

  compute_info = &it->second.compute_info;
  return Status::OK();
}

}  // namespace onnxruntime
