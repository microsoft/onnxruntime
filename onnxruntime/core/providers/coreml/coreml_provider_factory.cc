// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "coreml_execution_provider.h"
#include "coreml_provider_factory_creator.h"

using namespace onnxruntime;

namespace onnxruntime {

struct CoreMLProviderFactory : IExecutionProviderFactory {
  CoreMLProviderFactory(const CoreMLOptions& options)
      : options_(options) {}
  ~CoreMLProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  CoreMLOptions options_;
};

std::unique_ptr<IExecutionProvider> CoreMLProviderFactory::CreateProvider() {
  return std::make_unique<CoreMLExecutionProvider>(options_);
}

std::shared_ptr<IExecutionProviderFactory> CoreMLProviderFactoryCreator::Create(uint32_t coreml_flags) {
  CoreMLOptions coreml_options(coreml_flags);
  return std::make_shared<onnxruntime::CoreMLProviderFactory>(coreml_options);
}

std::shared_ptr<IExecutionProviderFactory> CoreMLProviderFactoryCreator::Create(const ProviderOptions& options) {
  CoreMLOptions coreml_options(options);
  return std::make_shared<onnxruntime::CoreMLProviderFactory>(coreml_options);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CoreML,
                    _In_ OrtSessionOptions* options, uint32_t coreml_flags) {
  options->provider_factories.push_back(onnxruntime::CoreMLProviderFactoryCreator::Create(coreml_flags));
  return nullptr;
}
