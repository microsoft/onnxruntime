// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/intel/intel_provider_factory.h"
#include "intel_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {
struct IntelProviderFactory : IExecutionProviderFactory {
  IntelProviderFactory(const char* device) : device_(device) {
  }
  ~IntelProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  const char* device_;
};

std::unique_ptr<IExecutionProvider> IntelProviderFactory::CreateProvider() {
  IntelExecutionProviderInfo info;
  //info.device = device_;
  return std::make_unique<IntelExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Intel(
    const char* device_id) {
  return std::make_shared<onnxruntime::IntelProviderFactory>(device_id);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Intel,
                    _In_ OrtSessionOptions* options, const char* device_id) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_Intel(device_id));
  return nullptr;
}
