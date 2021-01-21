// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "coreml_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {
struct CoreMLProviderFactory : IExecutionProviderFactory {
  CoreMLProviderFactory() {}
  ~CoreMLProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> CoreMLProviderFactory::CreateProvider() {
  return onnxruntime::make_unique<CoreMLExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CoreML() {
  return std::make_shared<onnxruntime::CoreMLProviderFactory>();
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CoreML, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CoreML());
  return nullptr;
}
