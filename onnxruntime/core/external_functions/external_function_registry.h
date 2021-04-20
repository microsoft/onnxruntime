// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace external_functions {

using ExternalFunctionMap = std::unordered_map<std::string, std::pair<void*, void*>>;

class ExternalFunctionRegistry final {
 public:
  static ExternalFunctionRegistry& GetInstance() {
    static ExternalFunctionRegistry instance_;
    return instance_;
  }

  void Register(const std::string& function_name, void* p_forward_fn, void* p_backward_fn) {
    external_functions_.insert({function_name, {p_forward_fn, p_backward_fn}});
  }

  void* GetFunction(const std::string& function_name, bool is_backward) {
    auto it = external_functions_.find(function_name);
    return it != external_functions_.end() ? (is_backward ? it->second.second : it->second.first) : nullptr;
  }

 private:
  ExternalFunctionRegistry() = default;
  ~ExternalFunctionRegistry() = default;
  ExternalFunctionRegistry(const ExternalFunctionRegistry&) = delete;
  ExternalFunctionRegistry& operator=(const ExternalFunctionRegistry&) = delete;

  ExternalFunctionMap external_functions_;
};

template <class T>
std::unique_ptr<OpKernel> CreateExternalFunctionKernel(const OpKernelInfo& info, void* p_fn_raw);

#define SPECIALIZED_EXTERNAL_FUNCTION_KERNEL_CREATOR(fn_name)                                                 \
  template <>                                                                                                 \
  std::unique_ptr<OpKernel> CreateExternalFunctionKernel<fn_name>(const OpKernelInfo& info, void* p_fn_raw) { \
    return std::make_unique<fn_name>(info, p_fn_raw);                                                         \
  }

std::unique_ptr<OpKernel> CreateExternalFunctionKernel(const std::string& name, const OpKernelInfo& info,
                                                       bool is_backward);

}  // namespace external_functions
}  // namespace onnxruntime
