// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <vector>
#include <list>
#include <mutex>
#include "core/common/status.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
struct KernelCreateInfo;
class ExecutionProviders;
class IExecutionProvider;
class KernelRegistry;
class OpKernel;
class SessionState;

enum class KernelRegistryPriority {
  HighPriority,
  LowPriority
};

// Kernel registries' manager.
// There're 2 kinds of kernel registries with priority from high to low as below,
// 1. Custom execution provider type specific kernel registries.
// 2. common execution provider type specific kernel registries.
// The 1st and 2nd ones are shared across sessions.
class KernelRegistryManager {
 public:
  KernelRegistryManager() = default;

  void RegisterKernels(const ExecutionProviders& execution_providers,
                       KernelRegistryPriority priority = KernelRegistryPriority::LowPriority);

  void RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry, KernelRegistryPriority priority);

  Status CreateKernel(const onnxruntime::Node& node,
                      const IExecutionProvider& execution_provider,
                      const SessionState& session_state,
                      /*out*/ std::unique_ptr<OpKernel>& op_kernel) const;

  Status SearchKernelRegistry(const onnxruntime::Node& node,
                              /*out*/ const KernelCreateInfo** kernel_create_info) const;

  // Get all kernel registries. There are no nullptr entries.
  std::vector<const KernelRegistry*> GetAllKernelRegistries() const {
    std::vector<const KernelRegistry*> result;
    for (auto& registry : kernel_registries_) {
      result.push_back(registry.get());
    }
    return result;
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelRegistryManager);

  // This list stores all kernel registries shared across sessions, including common ones and customized ones.
  std::list<std::shared_ptr<KernelRegistry>> kernel_registries_;
  mutable std::mutex lock_;
};
}  // namespace onnxruntime
