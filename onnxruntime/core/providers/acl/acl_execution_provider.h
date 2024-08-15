// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "core/platform/threadpool.h"

#include "arm_compute/runtime/MemoryManagerOnDemand.h"

namespace onnxruntime {

// Information needed to construct ACL execution providers.
struct ACLExecutionProviderInfo {
  bool enable_fast_math{false};

  explicit ACLExecutionProviderInfo(bool enable_fast_math)
      : enable_fast_math(enable_fast_math) {}

  ACLExecutionProviderInfo() = default;
};

// Logical device representation.
class ACLExecutionProvider : public IExecutionProvider {
 public:
  explicit ACLExecutionProvider(const ACLExecutionProviderInfo& info);
  virtual ~ACLExecutionProvider();

  const void* GetExecutionHandle() const noexcept override {
    // The ACL interface does not return anything interesting.
    return nullptr;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const IKernelLookup& kernel_lookup) const override;

  Status OnRunStart(const onnxruntime::RunOptions&) override;

  Status OnRunEnd(bool, const onnxruntime::RunOptions&) override;

  void SetThreadPool(concurrency::ThreadPool* thread_pool) {
    thread_pool_ = thread_pool;
  }

  concurrency::ThreadPool* GetThreadPool() const {
    return thread_pool_;
  }

  const ACLExecutionProviderInfo info;
  const std::shared_ptr<arm_compute::MemoryManagerOnDemand> memory_manager;
  concurrency::ThreadPool* thread_pool_ = nullptr;
};

}  // namespace onnxruntime
