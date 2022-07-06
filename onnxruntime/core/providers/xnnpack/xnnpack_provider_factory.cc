// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/xnnpack_provider_factory.h"

#include "core/providers/xnnpack/xnnpack_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct XnnpackProviderFactory : IExecutionProviderFactory {
  XnnpackProviderFactory() = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> XnnpackProviderFactory::CreateProvider() {
  XnnpackExecutionProviderInfo info{true};
  return std::make_unique<XnnpackExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Xnnpack() {
  return std::make_shared<XnnpackProviderFactory>();
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Xnnpack, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Xnnpack());
  return nullptr;
}
