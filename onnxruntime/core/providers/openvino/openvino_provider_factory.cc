// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "openvino_execution_provider.h"
#include "openvino_provider_factory_creator.h"

namespace onnxruntime {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(const char* device_type, bool enable_vpu_fast_compile,
                          const char* device_id, size_t num_of_threads,
                          const char* cache_dir, void* context,
                          bool enable_opencl_throttling, bool enable_dynamic_shapes)
      : enable_vpu_fast_compile_(enable_vpu_fast_compile), num_of_threads_(num_of_threads), context_(context), enable_opencl_throttling_(enable_opencl_throttling), enable_dynamic_shapes_(enable_dynamic_shapes) {
    device_type_ = (device_type == nullptr) ? "" : device_type;
    device_id_ = (device_id == nullptr) ? "" : device_id;
    cache_dir_ = (cache_dir == nullptr) ? "" : cache_dir;
  }
  ~OpenVINOProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  std::string device_type_;
  bool enable_vpu_fast_compile_;
  std::string device_id_;
  size_t num_of_threads_;
  std::string cache_dir_;
  void* context_;
  bool enable_opencl_throttling_;
  bool enable_dynamic_shapes_;
};

std::unique_ptr<IExecutionProvider> OpenVINOProviderFactory::CreateProvider() {
  OpenVINOExecutionProviderInfo info(device_type_, enable_vpu_fast_compile_, device_id_, num_of_threads_,
                                     cache_dir_, context_, enable_opencl_throttling_,
                                     enable_dynamic_shapes_);
  return std::make_unique<OpenVINOExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
    const char* device_type, bool enable_vpu_fast_compile, const char* device_id, size_t num_of_threads,
    const char* cache_dir, void* context, bool enable_opencl_throttling,
    bool enable_dynamic_shapes) {
  return std::make_shared<onnxruntime::OpenVINOProviderFactory>(device_type, enable_vpu_fast_compile,
                                                                device_id, num_of_threads, cache_dir, context, enable_opencl_throttling,
                                                                enable_dynamic_shapes);
}

}  // namespace onnxruntime

namespace onnxruntime {
struct ProviderInfo_OpenVINO_Impl : ProviderInfo_OpenVINO {
  std::vector<std::string> GetAvailableDevices() const override {
    openvino_ep::OVCore ie_core;
    return ie_core.GetAvailableDevices();
  }
} g_info;

struct OpenVINO_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    auto& params = *reinterpret_cast<const OrtOpenVINOProviderOptions*>(void_params);
    return std::make_shared<OpenVINOProviderFactory>(params.device_type, params.enable_vpu_fast_compile,
                                                     params.device_id, params.num_of_threads,
                                                     params.cache_dir,
                                                     params.context, params.enable_opencl_throttling,
                                                     params.enable_dynamic_shapes);
  }

  void Initialize() override {
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
