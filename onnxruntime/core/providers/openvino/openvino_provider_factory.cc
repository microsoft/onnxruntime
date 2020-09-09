// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/openvino_provider_factory.h"
#include "openvino_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(const char* device, bool enable_vpu_fast_compile)
    : enable_vpu_fast_compile_(enable_vpu_fast_compile) {
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
  std::string device_;
  bool enable_vpu_fast_compile_;
};

std::unique_ptr<IExecutionProvider> OpenVINOProviderFactory::CreateProvider() {
  OpenVINOExecutionProviderInfo info(device_, enable_vpu_fast_compile_);
  return std::make_unique<OpenVINOExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
    const char* device_id, bool enable_vpu_fast_compile) {
  return std::make_shared<onnxruntime::OpenVINOProviderFactory>(device_id, enable_vpu_fast_compile);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, const char* device_id, bool enable_vpu_fast_compile) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_OpenVINO(device_id, enable_vpu_fast_compile));
  return nullptr;
}
