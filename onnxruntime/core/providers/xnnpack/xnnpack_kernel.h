// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/xnnpack/xnnpack_execution_provider.h"
#include "xnnpack/cache.h"

struct pthreadpool;

namespace onnxruntime {
namespace xnnpack {

class XnnpackKernel : public OpKernel {
 public:
  explicit XnnpackKernel(const OpKernelInfo& info, bool enable_caches = false)
      : OpKernel{info},
        xnnpack_threadpool_{
            static_cast<const XnnpackExecutionProvider*>(info.GetExecutionProvider())->GetPrivateThreadPool()},
        caches_{enable_caches} {
  }
  [[nodiscard]] pthreadpool* GetThreadPool() const {
    return xnnpack_threadpool_;
  }

  xnn_code_cache_t GetCodeCache() { return caches_.auto_code_cache.get(); }
  xnn_weights_cache_t GetWeightsCache() { return caches_.auto_weights_cache.get(); }

 private:
  pthreadpool* xnnpack_threadpool_;

  struct Caches {
    Caches(bool enable)
        : auto_code_cache(nullptr, xnn_release_code_cache),
          auto_weights_cache(nullptr, xnn_release_weights_cache) {
      if (enable) {
#ifdef XNN_CACHE_ENABLE
        xnn_status status = xnn_status_success;
#if XNN_PLATFORM_JIT
        status = xnn_init_code_cache(&code_cache_);
        ORT_ENFORCE(status == xnn_status_success, "Failed to initialize XNNPACK code cache");)
        auto_code_cache.reset(&code_cache_);
#endif
        status = xnn_init_weights_cache(&weights_cache_);
        ORT_ENFORCE(status == xnn_status_success, "Failed to initialize XNNPACK weights cache");
        auto_weights_cache.reset(&weights_cache_);
#endif
      }
    }

    std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_code_cache;
    std::unique_ptr<xnn_weights_cache, decltype(&xnn_release_weights_cache)> auto_weights_cache;

   private:
#ifdef XNN_CACHE_ENABLE
#if XNN_PLATFORM_JIT
    xnn_code_cache code_cache_;
#endif
    xnn_weights_cache weights_cache_;
#endif
  };

  Caches caches_;
};
}  // namespace xnnpack
}  // namespace onnxruntime
