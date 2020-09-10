// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/openvino_provider_factory.h"
#include "openvino_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(const char* device_type, bool enable_vpu_fast_compile,
                          const char* device_id)
    : enable_vpu_fast_compile_(enable_vpu_fast_compile) {
    device_type_ = (device_type == nullptr) ? "" : device_type;
    device_id_ = (device_id == nullptr) ? "" : device_id;
  }
  ~OpenVINOProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  std::string device_type_;
  bool enable_vpu_fast_compile_;
  std::string device_id_;
};

std::unique_ptr<IExecutionProvider> OpenVINOProviderFactory::CreateProvider() {
  OpenVINOExecutionProviderInfo info(device_type_, enable_vpu_fast_compile_, device_id_);
  return std::make_unique<OpenVINOExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
    const char* device_type, bool enable_vpu_fast_compile, const char* device_id) {
  return std::make_shared<onnxruntime::OpenVINOProviderFactory>(device_type, enable_vpu_fast_compile, device_id);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, const char* device_type,
                    bool enable_vpu_fast_compile, const char* device_id) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_OpenVINO(device_type, enable_vpu_fast_compile, device_id));
  return nullptr;
}
