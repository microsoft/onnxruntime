// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"
#include "core/framework/session_options.h"

struct pthreadpool;
namespace onnxruntime {
// placeholder for future use. no options currently
struct XnnpackExecutionProviderInfo {
  int xnn_thread_pool_size{0};
  const SessionOptions* session_options{nullptr};
  XnnpackExecutionProviderInfo() = default;

  XnnpackExecutionProviderInfo(const ProviderOptions& po, const SessionOptions* sess_option)
      : session_options(sess_option) {
    if (auto it = po.find("intra_op_num_threads"); it != po.end()) {
      xnn_thread_pool_size = std::stoi(it->second);
    }
  }
};

class XnnpackExecutionProvider : public IExecutionProvider {
 public:
  XnnpackExecutionProvider(const XnnpackExecutionProviderInfo& info);
  ~XnnpackExecutionProvider() override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph_viewer,
      const IKernelLookup& /*kernel_lookup*/) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  DataLayout GetPreferredLayout() const override { return DataLayout::NHWC; }

  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

  // xnnpack does not support concurrent execution of a kernel
  bool ConcurrentRunSupported() const override { return false; }

  pthreadpool* GetPrivateThreadPool() const {
    return xnnpack_thread_pool_;
  }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

 private:
  pthreadpool* xnnpack_thread_pool_{nullptr};
};

}  // namespace onnxruntime
