// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/error_code_helper.h"
#include "core/providers/xnnpack/xnnpack_execution_provider.h"
#include "core/providers/xnnpack/xnnpack_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct XnnpackProviderFactory : IExecutionProviderFactory {
  XnnpackProviderFactory(const ProviderOptions& provider_options, const SessionOptions* session_options)
      : info_{provider_options, session_options} {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<XnnpackExecutionProvider>(info_);
  }

 private:
  XnnpackExecutionProviderInfo info_;
};

std::shared_ptr<IExecutionProviderFactory> XnnpackProviderFactoryCreator::Create(
    const ProviderOptions& provider_options, const SessionOptions* session_options) {
  return std::make_shared<XnnpackProviderFactory>(provider_options, session_options);
}

}  // namespace onnxruntime
