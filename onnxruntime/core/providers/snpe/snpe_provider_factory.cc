// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/snpe/snpe_provider_factory_creator.h"

#include "core/providers/snpe/snpe_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

namespace onnxruntime {
struct SNPEProviderFactory : IExecutionProviderFactory {
  explicit SNPEProviderFactory(const ProviderOptions& provider_options_map)
      : provider_options_map_(provider_options_map) {
  }
  ~SNPEProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<SNPEExecutionProvider>(provider_options_map_);
  }

 private:
  ProviderOptions provider_options_map_;
};

std::shared_ptr<IExecutionProviderFactory>
SNPEProviderFactoryCreator::Create(const ProviderOptions& provider_options_map) {
  return std::make_shared<onnxruntime::SNPEProviderFactory>(provider_options_map);
}

}  // namespace onnxruntime
