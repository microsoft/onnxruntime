// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "coreml_execution_provider.h"
#include "coreml_provider_factory_creator.h"

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
  return std::make_unique<CoreMLExecutionProvider>(coreml_flags_);
}

std::shared_ptr<IExecutionProviderFactory> CoreMLProviderFactoryCreator::Create(uint32_t coreml_flags) {
  return std::make_shared<onnxruntime::CoreMLProviderFactory>(coreml_flags);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CoreML,
                    _In_ OrtSessionOptions* options, uint32_t coreml_flags) {
  options->provider_factories.push_back(onnxruntime::CoreMLProviderFactoryCreator::Create(coreml_flags));
  return nullptr;
}
