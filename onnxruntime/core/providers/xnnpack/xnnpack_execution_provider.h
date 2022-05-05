// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"

namespace onnxruntime {
struct SessionOptions;

struct XnnpackExecutionProviderInfo {
  const SessionOptions* session_options{nullptr};  // required if you want fusion of Conv+Activation
  bool create_arena{true};

  explicit XnnpackExecutionProviderInfo(const SessionOptions* so = nullptr, bool use_arena = true)
      : session_options{so}, create_arena{use_arena} {}

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

  DataLayout GetPreferredLayout() const override { return DataLayout::NHWC; }

  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

 private:
  const SessionOptions* session_options_{nullptr};
};

}  // namespace onnxruntime
