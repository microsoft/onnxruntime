// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
/**
 * Each provider has a KernelRegistry. Often, the KernelRegistry only belongs to that specific provider.
 *
 */
class KernelRegistry {
 public:
  KernelRegistry() = default;

  // Register a kernel with kernel definition and function to create the kernel.
  Status Register(KernelDefBuilder& kernel_def_builder, const KernelCreateFn& kernel_creator) ORT_MUST_USE_RESULT;

  Status Register(KernelCreateInfo&& create_info) ORT_MUST_USE_RESULT;

#if !defined(ORT_MINIMAL_BUILD)
  static bool HasImplementationOf(const KernelRegistry& r, const onnxruntime::Node& node,
                                  onnxruntime::ProviderType exec_provider) {
    const KernelCreateInfo* info;
    Status st = r.TryFindKernel(node, exec_provider, &info);
    return st.IsOK();
  }

  // factory functions should always return a unique_ptr for maximum flexibility
  // for its clients unless the factory is managing the lifecycle of the pointer
  // itself.
  // TODO(Task:132) Make usage of unique_ptr/shared_ptr as out param consistent
  Status TryCreateKernel(const onnxruntime::Node& node, const IExecutionProvider& execution_provider,
                         const std::unordered_map<int, OrtValue>& constant_initialized_tensors,
                         const OrtValueNameIdxMap& mlvalue_name_idx_map, const FuncManager& funcs_mgr,
                         const DataTransferManager& data_transfer_mgr,
                         std::unique_ptr<OpKernel>& op_kernel) const ORT_MUST_USE_RESULT;

  // Check if an execution provider can create kernel for a node and return the kernel if so
  Status TryFindKernel(const onnxruntime::Node& node, onnxruntime::ProviderType exec_provider,
                       const KernelCreateInfo** out) const;

#endif

  // Check if an execution provider can create kernel for a node and return the kernel if so.
  // Kernel matching is via kernel_def_hash.
  Status TryFindKernel(const onnxruntime::Node& node, onnxruntime::ProviderType exec_provider,
                       uint64_t kernel_def_hash,
                       const KernelCreateInfo** out) const;

  bool IsEmpty() const { return kernel_creator_fn_map_.empty(); }

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA
  // This is used by the opkernel doc generator to enlist all registered operators for a given provider's opkernel
  const KernelCreateMap& GetKernelCreateMap() const {
    return kernel_creator_fn_map_;
  }
#endif

 private:
#if !defined(ORT_MINIMAL_BUILD)
  // Check whether the types of inputs/outputs of the given node match the extra
  // type-constraints of the given kernel. This serves two purposes: first, to
  // select the right kernel implementation based on the types of the arguments
  // when we have multiple kernels, e.g., Clip<float> and Clip<int>; second, to
  // accommodate (and check) mapping of ONNX (specification) type to the onnxruntime
  // implementation type (e.g., if we want to implement ONNX's float16 as a regular
  // float in onnxruntime). (The second, however, requires a globally uniform mapping.)
  //
  // Note that this is not intended for type-checking the node against the ONNX
  // type specification of the corresponding op, which is done before this check.
  //
  // if this function is called before graph partition, then node.provider is not set.
  // In this case, kernel_def.provider must equal to exec_provider
  // otherwise, kernel_def.provider must equal to node.provider. exec_provider is ignored.
  static bool VerifyKernelDef(const onnxruntime::Node& node,
                              const KernelDef& kernel_def,
                              std::string& error_str);
#endif

  static std::string GetMapKey(const std::string& op_name, const std::string& domain, const std::string& provider) {
    std::string key(op_name);
    key.append(1, ' ').append(domain.empty() ? kOnnxDomainAlias : domain).append(1, ' ').append(provider);
    return key;
  }

  static std::string GetMapKey(const KernelDef& kernel_def) {
    return GetMapKey(kernel_def.OpName(), kernel_def.Domain(), kernel_def.Provider());
  }
  // Kernel create function map from op name to kernel creation info.
  // key is opname+domain_name+provider_name
  KernelCreateMap kernel_creator_fn_map_;
};
}  // namespace onnxruntime
