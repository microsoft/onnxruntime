// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/external_functions/external_function_registry.h"

namespace onnxruntime {
namespace external_functions {

#define EXTERNAL_FUNCTION_KERNEL_CREATOR_INFO(name, forward_fn, backward_fn)                      \
  {                                                                                               \
    name, { CreateExternalFunctionKernel<forward_fn>, CreateExternalFunctionKernel<backward_fn> } \
  }

class ATenEmbeddingFunction;
class ATenEmbeddingBackwardFunction;

using ExternalFunctionKernelFnCreator = std::unique_ptr<OpKernel> (*)(const OpKernelInfo&, void*);
using ExternalFunctionKernelFnCreatorMap =
    std::unordered_map<std::string, std::pair<ExternalFunctionKernelFnCreator, ExternalFunctionKernelFnCreator>>;

std::unique_ptr<OpKernel> CreateExternalFunctionKernel(const std::string& name, const OpKernelInfo& info,
                                                       bool is_backward) {
  static const ExternalFunctionKernelFnCreatorMap creator_map = {
      EXTERNAL_FUNCTION_KERNEL_CREATOR_INFO("aten::embedding", ATenEmbeddingFunction, ATenEmbeddingBackwardFunction),
  };

  void* p_fn_raw = ExternalFunctionRegistry::GetInstance().GetFunction(name, is_backward);
  auto it = creator_map.find(name);
  if (!p_fn_raw || it == creator_map.end()) {
    return std::unique_ptr<OpKernel>(nullptr);
  }

  ExternalFunctionKernelFnCreator fn_creator = is_backward ? it->second.second : it->second.first;
  return fn_creator(info, p_fn_raw);
}

}  // namespace external_functions
}  // namespace onnxruntime
