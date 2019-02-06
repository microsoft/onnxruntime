// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
class KernelRegistry {
 public:
  KernelRegistry() = default;

  // Register a kernel with kernel definition and function to create the kernel.
  Status Register(KernelDefBuilder& kernel_def_builder,
                  const KernelCreateFn& kernel_creator);

  Status Register(KernelCreateInfo&& create_info);

  // Mainly for provide debug info
  std::vector<std::string> GetAllRegisteredOpNames() const;

  // factory functions should always return a unique_ptr for maximum flexibility
  // for its clients unless the factory is managing the lifecycle of the pointer
  // itself.
  // TODO(Task:132) Make usage of unique_ptr/shared_ptr as out param consistent
  Status CreateKernel(const onnxruntime::Node& node,
                      const IExecutionProvider& execution_provider,
                      const std::unordered_map<int, MLValue>& initialized_tensors,
                      const MLValueNameIdxMap& mlvalue_name_idx_map,
                      const FuncManager& funcs_mgr,
                      std::unique_ptr<OpKernel>& op_kernel) const;

  // Check if an execution provider can create kernel for a node and return
  // the kernel if so
  const KernelCreateInfo* TryFindKernel(const onnxruntime::Node& node,
                                        onnxruntime::ProviderType exec_provider) const;

 private:
  // Check if the node's input/outpuData/attributes are compatible with this
  // kernel_def, If so, the kernel defined by the kernel_def is used to
  // execute this node. exec_provider is used to match kernel when node has no provider
  static bool VerifyKernelDef(const onnxruntime::Node& node,
                              const KernelDef& kernel_def,
                              std::string& error_str,
                              onnxruntime::ProviderType exec_provider = "");

  // Kernel create function map from op name to kernel creation info.
  KernelCreateMap kernel_creator_fn_map_;
};
}  // namespace onnxruntime
