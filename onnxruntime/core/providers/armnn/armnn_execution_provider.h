// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct ArmNN execution providers.
struct ArmNNExecutionProviderInfo {
  bool create_arena{true};

  explicit ArmNNExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  ArmNNExecutionProviderInfo() = default;
};

// Logical device representation.
class ArmNNExecutionProvider : public IExecutionProvider {
 public:
  explicit ArmNNExecutionProvider(const ArmNNExecutionProviderInfo& info);
  virtual ~ArmNNExecutionProvider();

  const void* GetExecutionHandle() const noexcept override {
    // The ArmNN interface does not return anything interesting.
    return nullptr;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace onnxruntime
