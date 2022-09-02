// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"
#include "xnnpack.h"

namespace onnxruntime {
// placeholder for future use. no options currently
struct XnnpackExecutionProviderInfo {
  int xnn_thread_pool_size{0};
  XnnpackExecutionProviderInfo() = default;

  XnnpackExecutionProviderInfo(const ProviderOptions& po) {
    if (po.count("thread_num")) {
      xnn_thread_pool_size = std::stoi(po.at("thread_num"));
    }
    // future: parse ProviderOptions
  };
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

  pthreadpool_t GetPrivateThreadPool() const {
    return xnnpack_thread_pool_.get();
  }

 private:
  // Thread pool with smart-pointer for lifetime management.
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> xnnpack_thread_pool_{
      nullptr, &pthreadpool_destroy};
};

}  // namespace onnxruntime
