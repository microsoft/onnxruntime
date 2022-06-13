// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"

namespace onnxruntime {
struct XnnpackExecutionProviderInfo {
  bool create_arena{true};

  explicit XnnpackExecutionProviderInfo(bool use_arena = true)
      : create_arena{use_arena} {}

  XnnpackExecutionProviderInfo() = delete;
};

class XnnpackExecutionProvider : public IExecutionProvider {
 public:
  XnnpackExecutionProvider(const XnnpackExecutionProviderInfo& info);
  ~XnnpackExecutionProvider() override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& kernel_registries) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  void RegisterAllocator(AllocatorManager& /*allocator_manager*/) override;

  DataLayout GetPreferredLayout() const override { return DataLayout::NHWC; }

  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

  // xnnpack does not support concurrent execution of a kernel
  bool ConcurrentRunSupported() const override { return false; }
};

}  // namespace onnxruntime
