// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  ProviderOptions provider_options_map_;
};

std::shared_ptr<IExecutionProviderFactory>
CreateExecutionProviderFactory_SNPE(const ProviderOptions& provider_options_map) {
  return std::make_shared<onnxruntime::SNPEProviderFactory>(provider_options_map);
}

std::unique_ptr<IExecutionProvider> SNPEProviderFactory::CreateProvider() {
  return std::make_unique<SNPEExecutionProvider>(provider_options_map_);
}

}  // namespace onnxruntime


ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_SNPE, _In_ OrtSessionOptions* options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    _In_ size_t num_keys) {
  API_IMPL_BEGIN
  onnxruntime::ProviderOptions provider_options_map;
  for (size_t i = 0; i != num_keys; ++i) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "key/value cannot be empty");
    }

    provider_options_map[provider_options_keys[i]] = provider_options_values[i];
  }
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_SNPE(provider_options_map));
  return nullptr;
  API_IMPL_END
}
