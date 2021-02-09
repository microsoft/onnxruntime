// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "coreml_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {
struct CoreMLProviderFactory : IExecutionProviderFactory {
  CoreMLProviderFactory(uint32_t coreml_flags)
      : coreml_flags_(coreml_flags) {}
  ~CoreMLProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  uint32_t coreml_flags_;
};

std::unique_ptr<IExecutionProvider> CoreMLProviderFactory::CreateProvider() {
  return onnxruntime::make_unique<CoreMLExecutionProvider>(coreml_flags_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CoreML(uint32_t coreml_flags) {
  return std::make_shared<onnxruntime::CoreMLProviderFactory>(coreml_flags);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CoreML,
                    _In_ OrtSessionOptions* options, uint32_t coreml_flags) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CoreML(coreml_flags));
  return nullptr;
}
