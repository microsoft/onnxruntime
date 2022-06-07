// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct ACL execution providers.
struct ACLExecutionProviderInfo {
  bool create_arena{true};

  explicit ACLExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  ACLExecutionProviderInfo() = default;
};

// Logical device representation.
class ACLExecutionProvider : public IExecutionProvider {
 public:
  explicit ACLExecutionProvider(const ACLExecutionProviderInfo& info);
  virtual ~ACLExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& kernel_registries) const override;

  const void* GetExecutionHandle() const noexcept override {
    // The ACL interface does not return anything interesting.
    return nullptr;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace onnxruntime
