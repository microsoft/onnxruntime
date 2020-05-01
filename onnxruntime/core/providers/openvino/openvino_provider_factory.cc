// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/openvino_provider_factory.h"
#include "openvino_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(const char* device) {
    if (device == nullptr) {
      device_ = "";
    } else {
      device_ = device;
    }
  }
  ~OpenVINOProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  const char* device_;
};

std::unique_ptr<IExecutionProvider> OpenVINOProviderFactory::CreateProvider() {
  OpenVINOExecutionProviderInfo info(device_);
  return std::make_unique<OpenVINOExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
    const char* device_id) {
  return std::make_shared<onnxruntime::OpenVINOProviderFactory>(device_id);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, const char* device_id) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_OpenVINO(device_id));
  return nullptr;
}
