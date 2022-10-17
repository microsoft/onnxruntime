// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

#include "core/framework/op_kernel.h"

namespace onnxruntime {

using KernelCreateMap = std::multimap<std::string, KernelCreateInfo>;
using KernelDefHashes = std::vector<std::pair<std::string, HashValue>>;

class IKernelTypeStrResolver;

/**
 * Each provider has a KernelRegistry. Often, the KernelRegistry only belongs to that specific provider.
 */
class KernelRegistry {
 public:
  KernelRegistry() = default;

  // Register a kernel with kernel definition and function to create the kernel.
  Status Register(KernelDefBuilder& kernel_def_builder, const KernelCreateFn& kernel_creator);

  Status Register(KernelCreateInfo&& create_info);

  // TODO(edgchen1) for TryFindKernel(), consider using `out` != nullptr as indicator of whether kernel was found and
  // Status as an indication of failure

  // Check if an execution provider can create kernel for a node and return the kernel if so
  Status TryFindKernel(const Node& node, ProviderType exec_provider,
                       const IKernelTypeStrResolver& kernel_type_str_resolver,
                       const KernelCreateInfo** out) const;

  static bool HasImplementationOf(const KernelRegistry& r, const Node& node,
                                  ProviderType exec_provider,
                                  const IKernelTypeStrResolver& kernel_type_str_resolver) {
    const KernelCreateInfo* info;
    Status st = r.TryFindKernel(node, exec_provider, kernel_type_str_resolver, &info);
    return st.IsOK();
  }

#if !defined(ORT_MINIMAL_BUILD)
  // Find KernelCreateInfo in instant mode
  Status TryFindKernel(const std::string& op_name, const std::string& domain, const int& version,
                       const std::unordered_map<std::string, MLDataType>& type_constraints,
                       ProviderType exec_provider, const KernelCreateInfo** out) const;
#endif  // !defined(ORT_MINIMAL_BUILD)

  bool IsEmpty() const { return kernel_creator_fn_map_.empty(); }

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA
  // This is used by the opkernel doc generator to enlist all registered operators for a given provider's opkernel
  const KernelCreateMap& GetKernelCreateMap() const {
    return kernel_creator_fn_map_;
  }
#endif

 private:
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
  static bool VerifyKernelDef(const Node& node,
                              const KernelDef& kernel_def,
                              const IKernelTypeStrResolver& kernel_type_str_resolver,
                              std::string& error_str);

  static std::string GetMapKey(std::string_view op_name, std::string_view domain, std::string_view provider) {
    std::string key(op_name);
    // use the kOnnxDomainAlias of 'ai.onnx' instead of kOnnxDomain's empty string
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
