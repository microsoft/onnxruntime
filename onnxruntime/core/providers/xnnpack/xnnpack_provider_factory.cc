// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/xnnpack_provider_factory.h"
#include "xnnpack_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct XNNPackProviderFactory : IExecutionProviderFactory {
  XNNPackProviderFactory() {}
  ~XNNPackProviderFactory() override {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> XNNPackProviderFactory::CreateProvider() {
  return std::make_unique<XNNPackExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_XNNPack() {
  return std::make_shared<onnxruntime::XNNPackProviderFactory>();
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_XNNPack, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_XNNPack());
  return nullptr;
}
