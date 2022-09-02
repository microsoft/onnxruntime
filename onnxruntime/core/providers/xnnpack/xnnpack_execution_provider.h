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
  int xnn_thread_pool_size;
  XnnpackExecutionProviderInfo() = default;

  XnnpackExecutionProviderInfo(const ProviderOptions&){
      // future: parse ProviderOptions
  };
};

class XnnpackThreadPool {
 public:
  XnnpackThreadPool() = default;

  void InitializeWithPoolSize(int64_t num_threads) {
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
    if (num_threads > 1) {
      threadpool_.reset(
          pthreadpool_create(static_cast<size_t>(num_threads)));
    }
    printf("%zd\n", pthreadpool_get_threads_count(threadpool_.get()));
#endif
  }
  pthreadpool_t Get() {
    return threadpool_.get();
  }

 private:
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
  // Thread pool with smart-pointer for lifetime management.
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_{
      nullptr, &pthreadpool_destroy};
#endif
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
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    return nullptr;
#else
    return xnnpack_thread_pool_->Get();
#endif
  }

 private:
  std::unique_ptr<XnnpackThreadPool> xnnpack_thread_pool_;
};

}  // namespace onnxruntime
