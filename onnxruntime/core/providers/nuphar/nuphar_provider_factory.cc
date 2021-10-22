// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/nuphar_provider_factory.h"
#include <atomic>
#include "nuphar_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
//#include "core/codegen/passes/utils/codegen_context.h"  // TODO: remove it

namespace onnxruntime {
struct NupharExecutionProviderFactory : IExecutionProviderFactory {
  NupharExecutionProviderFactory(bool allow_unaligned_buffers, const char* settings)
      : settings_(settings),
        allow_unaligned_buffers_(allow_unaligned_buffers) {}
  ~NupharExecutionProviderFactory() = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  std::string settings_;
  bool allow_unaligned_buffers_;
};

std::unique_ptr<IExecutionProvider> NupharExecutionProviderFactory::CreateProvider() {
  NupharExecutionProviderInfo info(allow_unaligned_buffers_, settings_);
  return std::make_unique<NupharExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(bool allow_unaligned_buffers, const char* settings) {
  return std::make_shared<onnxruntime::NupharExecutionProviderFactory>(allow_unaligned_buffers, settings);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nuphar, _In_ OrtSessionOptions* options, int allow_unaligned_buffers, _In_ const char* settings) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Nuphar(static_cast<bool>(allow_unaligned_buffers), settings));
  return nullptr;
}
