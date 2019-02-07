// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include "core/common/status.h"
#include "core/platform/ort_mutex.h"
#include "core/graph/graph_viewer.h"

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

//This class is thread safe
class KernelRegistryManager {
 public:
  KernelRegistryManager() = default;

  void RegisterKernels(const ExecutionProviders& execution_providers);

  // The registry passed in this function has highest priority than anything already in this KernelRegistryManager
  // For example, if you do:
  // RegisterKernels(providers)
  // RegisterKernelRegistry(A);
  // RegisterKernelRegistry(B);
  // Then B > A > providers
  void RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry);

  Status CreateKernel(const onnxruntime::Node& node,
                      const IExecutionProvider& execution_provider,
                      const SessionState& session_state,
                      /*out*/ std::unique_ptr<OpKernel>& op_kernel) const;

  //Don't call this function before graph partition and graph transforms are done
  //This function assumes every node is already assigned to an execution provider
  Status SearchKernelRegistry(const onnxruntime::Node& node,
                              /*out*/ const KernelCreateInfo** kernel_create_info) const;

  // Get all kernel registries. There are no nullptr entries.
  std::vector<const KernelRegistry*> GetKernelRegistriesByProviderType(const std::string& type) const {
    std::vector<const KernelRegistry*> result;
    for (auto& registry : kernel_registries_) {
      result.push_back(registry.get());
    }
    auto iter = provider_type_to_registry_.find(type);
    if (iter != provider_type_to_registry_.end())
      result.push_back(iter->second.get());
    return result;
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelRegistryManager);

 private:
  std::unordered_map<std::string, std::shared_ptr<KernelRegistry>> provider_type_to_registry_;
  std::list<std::shared_ptr<KernelRegistry>> kernel_registries_;
  mutable OrtMutex lock_;
};
}  // namespace onnxruntime
