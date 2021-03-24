// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "openvino_execution_provider.h"

namespace onnxruntime {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(const char* device_type, bool enable_vpu_fast_compile,
                          const char* device_id, size_t num_of_threads,
                          bool use_compiled_network, const char* blob_dump_path)
      : enable_vpu_fast_compile_(enable_vpu_fast_compile), num_of_threads_(num_of_threads), use_compiled_network_(use_compiled_network) {
    device_type_ = (device_type == nullptr) ? "" : device_type;
    device_id_ = (device_id == nullptr) ? "" : device_id;
    blob_dump_path_ = (blob_dump_path == nullptr) ? "" : blob_dump_path;
  }
  ~OpenVINOProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  std::string device_type_;
  bool enable_vpu_fast_compile_;
  std::string device_id_;
  size_t num_of_threads_;
  bool use_compiled_network_;
  std::string blob_dump_path_;
};

std::unique_ptr<IExecutionProvider> OpenVINOProviderFactory::CreateProvider() {
  OpenVINOExecutionProviderInfo info(device_type_, enable_vpu_fast_compile_, device_id_, num_of_threads_, use_compiled_network_, blob_dump_path_);
  return std::make_unique<OpenVINOExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
    const char* device_type, bool enable_vpu_fast_compile, const char* device_id, size_t num_of_threads, bool use_compiled_network, const char* blob_dump_path) {
  return std::make_shared<onnxruntime::OpenVINOProviderFactory>(device_type, enable_vpu_fast_compile, device_id, num_of_threads, use_compiled_network, blob_dump_path);
}

}  // namespace onnxruntime

namespace onnxruntime {
struct ProviderInfo_OpenVINO_Impl : ProviderInfo_OpenVINO {
  std::vector<std::string> GetAvailableDevices() const override {
    InferenceEngine::Core ie_core;
    return ie_core.GetAvailableDevices();
  }
} g_info;

struct OpenVINO_Provider : Provider {
  const void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    auto& params = *reinterpret_cast<const OrtOpenVINOProviderOptions*>(void_params);
    return std::make_shared<OpenVINOProviderFactory>(params.device_type, params.enable_vpu_fast_compile, params.device_id, params.num_of_threads, params.use_compiled_network, params.blob_dump_path);
  }

  void Shutdown() override {
    openvino_ep::BackendManager::ReleaseGlobalContext();
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
