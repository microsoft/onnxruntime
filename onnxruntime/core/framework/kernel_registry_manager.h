// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include "core/common/status.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/customregistry.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
struct KernelCreateInfo;
class ExecutionProviders;
class IExecutionProvider;
class KernelRegistry;
class OpKernel;
class SessionState;

// Kernel registries' manager.
// There're 2 kinds of kernel registries with priority from high to low as below,
// 1. Custom execution provider type specific kernel registries.
// 2. common execution provider type specific kernel registries.
// The 1st and 2nd ones are shared across sessions.

// This class is not thread safe.
class KernelRegistryManager {
 public:
  KernelRegistryManager() = default;

  // Register kernels from providers
  Status RegisterKernels(const ExecutionProviders& execution_providers) ORT_MUST_USE_RESULT;

  // The registry passed in this function has highest priority than anything already in this KernelRegistryManager,
  // and anything registered from RegisterKernels
  // For example, if you do:
  // RegisterKernels(providers)
  // RegisterKernelRegistry(A);
  // RegisterKernelRegistry(B);
  // Then B > A > providers
  void RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry);

  // This function assumes the node is already assigned to an execution provider
  // Don't call this function before graph partition is done
  Status CreateKernel(const onnxruntime::Node& node, const IExecutionProvider& execution_provider,
                      const SessionState& session_state,
                      /*out*/ std::unique_ptr<OpKernel>& op_kernel) const ORT_MUST_USE_RESULT;

  // This function assumes the node is already assigned to an execution provider
  // Don't call this function before graph partition is done
  Status SearchKernelRegistry(const onnxruntime::Node& node,
                              /*out*/ const KernelCreateInfo** kernel_create_info) const;

  /**
   * Whether this node can be run on this provider
   */
  static bool HasImplementationOf(const KernelRegistryManager& r, const Node& node, const std::string& provider_type);

  /**
   * Search kernel registry by provider type.
   * @param type provider type string
   * @return It returns all the possible results. The returned value may contain garbage that doesn't belong to
   *         this provider. Caller should do the filtering. The returned value won't have no nullptrs.
   */
  std::vector<const KernelRegistry*> GetKernelRegistriesByProviderType(const std::string& type) const {
    std::vector<const KernelRegistry*> result;
    for (auto& registry : custom_kernel_registries_) {
      result.push_back(registry.get());
    }
    auto iter = provider_type_to_registry_.find(type);
    if (iter != provider_type_to_registry_.end()) result.push_back(iter->second.get());
    return result;
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelRegistryManager);

 private:
  // key is provider type. Each kernel registry in this collection only belongs to one specific provider
  std::unordered_map<std::string, std::shared_ptr<KernelRegistry>> provider_type_to_registry_;
  // Each kernel registry may contain kernels from many different providers.
  // in order to search kernels from a specific provider, we have to iterate all its elements
  std::list<std::shared_ptr<KernelRegistry>> custom_kernel_registries_;
};
}  // namespace onnxruntime
