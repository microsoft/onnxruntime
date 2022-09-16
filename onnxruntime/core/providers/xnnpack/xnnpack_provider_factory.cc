// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/error_code_helper.h"
#include "core/providers/xnnpack/xnnpack_execution_provider.h"
#include "core/providers/xnnpack/xnnpack_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct XnnpackProviderFactory : IExecutionProviderFactory {
  XnnpackProviderFactory(const ProviderOptions& provider_options)
      : info_{provider_options} {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider(const SessionOptions* options = nullptr) override {
    if (options && options->intra_op_param.allow_spinning && options->intra_op_param.thread_pool_size > 1) {
      LOGS_DEFAULT(WARNING)
          << "XNNPACK EP utilize pthreadpool for multi-threading. So, if allow_spinning on ORT's"
             "thread-pool and its pool size is not 1, "
             "pthreadpool will content with ORT's intra-op thread pool and hurt performance a lot. "
             "Please Setting intra_op_param.allow_spinning to false or "
             "setting ort's pool size (intra_thread_num) to 1 and try again.";
    }
    if (options && info_.xnn_thread_pool_size == 0) {
      LOGS_DEFAULT(WARNING) << "XNNPACK pool size is not set. Using ORT's thread-pool size as default:";
      info_.xnn_thread_pool_size = options->intra_op_param.thread_pool_size;
    }
    return std::make_unique<XnnpackExecutionProvider>(info_);
  }

 private:
  XnnpackExecutionProviderInfo info_;
};

std::shared_ptr<IExecutionProviderFactory> XnnpackProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  return std::make_shared<XnnpackProviderFactory>(provider_options);
}

}  // namespace onnxruntime
