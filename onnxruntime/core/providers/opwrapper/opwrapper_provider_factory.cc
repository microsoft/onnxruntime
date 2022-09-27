// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opwrapper/opwrapper_provider_factory_creator.h"

#include "core/providers/opwrapper/opwrapper_execution_provider.h"

namespace onnxruntime {
struct OpWrapperProviderFactory : IExecutionProviderFactory {
  explicit OpWrapperProviderFactory(const ProviderOptions& provider_options)
    : provider_options_(provider_options) {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<OpWrapperExecutionProvider>(provider_options_);
  }
 private:
  ProviderOptions provider_options_;
};

std::shared_ptr<IExecutionProviderFactory>
OpWrapperProviderFactoryCreator::Create(const ProviderOptions& provider_options) {
  return std::make_shared<onnxruntime::OpWrapperProviderFactory>(provider_options);
}

}  // namespace onnxruntime
