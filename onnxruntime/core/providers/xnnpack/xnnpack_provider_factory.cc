// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/xnnpack_provider_factory.h"

#include "core/providers/xnnpack/xnnpack_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct XnnpackProviderFactory : IExecutionProviderFactory {
  XnnpackProviderFactory(const SessionOptions* so) : so_{so} {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  const SessionOptions* so_;
};

std::unique_ptr<IExecutionProvider> XnnpackProviderFactory::CreateProvider() {
  XnnpackExecutionProviderInfo info{so_, true};
  return std::make_unique<XnnpackExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Xnnpack(const SessionOptions* so = nullptr) {
  return std::make_shared<onnxruntime::XnnpackProviderFactory>(so);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Xnnpack, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Xnnpack(&options->value));
  return nullptr;
}
