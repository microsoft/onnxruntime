// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/xnnpack/xnnpack_execution_provider.h"
#include "xnnpack.h"

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

  // see comment below about enabling code cache
  xnn_weights_cache_t GetWeightsCache() { return caches_.auto_weights_cache.get(); }

 private:
  pthreadpool* xnnpack_threadpool_;

  // Helper class to wrap usage of the XNNPACK weights and code caches.
  // NOTE: Currently creating/freeing the code cache is not exposed via the public xnnpack.h header so usage is
  // commented out. If we need to use it, we'll need to add the 'src' directory of XNNPACK to the include path
  // and #include "xnnpack/cache.h"
  struct Caches {
    Caches(bool enable)
        :  // auto_code_cache(nullptr, xnn_release_code_cache),
          auto_weights_cache(nullptr, xnn_delete_weights_cache) {
      if (enable) {
#ifdef XNN_CACHE_ENABLE
        xnn_status status = xnn_status_success;
        // status = xnn_init_weights_cache(&weights_cache_);
        xnn_weights_cache_t weights_cache_provider = nullptr;
        status = xnn_create_weights_cache(&weights_cache, 0);
        ORT_ENFORCE(status == xnn_status_success, "Failed to create XNNPACK weights cache");
        auto_weights_cache.reset(weights_cache);
#endif
      }
    }

    // std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_code_cache;
    std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache;

    // private:
    // #if defined(XNN_CACHE_ENABLE) && XNN_PLATFORM_JIT
    //   xnn_code_cache code_cache_;
    // #endif
  };

  Caches caches_;
};
}  // namespace xnnpack
}  // namespace onnxruntime
