// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#include <atomic>
#include "tensorrt_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct TensorrtProviderFactory : IExecutionProviderFactory {
  TensorrtProviderFactory() {}
  ~TensorrtProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> TensorrtProviderFactory::CreateProvider() {
  return std::make_unique<TensorrtExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt() {
  return std::make_shared<onnxruntime::TensorrtProviderFactory>();
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Tensorrt());
  return nullptr;
}

