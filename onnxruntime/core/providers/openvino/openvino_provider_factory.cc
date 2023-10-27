// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"

namespace onnxruntime {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(const char* device_type, bool enable_npu_fast_compile,
                          const char* device_id, size_t num_of_threads,
                          const char* cache_dir, int num_streams, void* context,
                          bool enable_opencl_throttling, bool enable_dynamic_shapes)
      : enable_npu_fast_compile_(enable_npu_fast_compile),
        num_of_threads_(num_of_threads),
        num_streams_(num_streams),
        context_(context),
        enable_opencl_throttling_(enable_opencl_throttling),
        enable_dynamic_shapes_(enable_dynamic_shapes) {
    device_type_ = (device_type == nullptr) ? "" : device_type;
    device_id_ = (device_id == nullptr) ? "" : device_id;
    cache_dir_ = (cache_dir == nullptr) ? "" : cache_dir;
  }
  ~OpenVINOProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  std::string device_type_;
  bool enable_npu_fast_compile_;
  std::string device_id_;
  size_t num_of_threads_;
  std::string cache_dir_;
  int num_streams_;
  void* context_;
  bool enable_opencl_throttling_;
  bool enable_dynamic_shapes_;
};

std::unique_ptr<IExecutionProvider> OpenVINOProviderFactory::CreateProvider() {
  OpenVINOExecutionProviderInfo info(device_type_, enable_npu_fast_compile_, device_id_, num_of_threads_,
                                     cache_dir_, num_streams_, context_, enable_opencl_throttling_,
                                     enable_dynamic_shapes_);
  return std::make_unique<OpenVINOExecutionProvider>(info);
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
    auto& provider_options_map = *reinterpret_cast<const ProviderOptions*>(void_params);

    std::string device_type = "";           // [device_type]: Overrides the accelerator hardware type and precision
                                            //   with these values at runtime.
    bool enable_npu_fast_compile = false;   // [enable_npu_fast_compile]: Fast-compile may be optionally enabled to
                                            // speeds up the model's compilation to NPU device specific format.
    const char* device_id = "";             // [device_id]: Selects a particular hardware device for inference.
    int num_of_threads = 8;                 // [num_of_threads]: Overrides the accelerator default value of number of
                                            //  threads with this value at runtime.
    const char* cache_dir = "";             // [cache_dir]: specify the path to
                                            // dump and load the blobs for the model caching/kernel caching (GPU)
                                            // feature. If blob files are already present, it will be directly loaded.
    int num_streams = 1;                    // [num_streams]: Option that specifies the number of parallel inference
                                            // requests to be processed on a given `device_type`. Overrides the
                                            // accelerator default value of number of streams
                                            // with this value at runtime.
    bool enable_opencl_throttling = false;  // [enable_opencl_throttling]: Enables OpenCL queue throttling for GPU
                                            // device (Reduces CPU Utilization when using GPU)
    bool enable_dynamic_shapes = false;     // [enable_dynamic_shapes]: Enables Dynamic Shapes feature for CPU device)
    void* context = nullptr;

    if (provider_options_map.find("device_type") != provider_options_map.end()) {
      device_type = provider_options_map.at("device_type").c_str();

      std::set<std::string> ov_supported_device_types = {"CPU_FP32", "CPU_FP16", "GPU_FP32",
                                                         "GPU.0_FP32", "GPU.1_FP32", "GPU_FP16",
                                                         "GPU.0_FP16", "GPU.1_FP16"};
      if (!((ov_supported_device_types.find(device_type) != ov_supported_device_types.end()) ||
            (device_type.find("HETERO:") == 0) ||
            (device_type.find("MULTI:") == 0) ||
            (device_type.find("AUTO:") == 0))) {
        ORT_THROW(
            "[ERROR] [OpenVINO] You have selcted wrong configuration value for the key 'device_type'. "
            "Select from 'CPU_FP32', 'CPU_FP16', 'GPU_FP32', 'GPU.0_FP32', 'GPU.1_FP32', 'GPU_FP16', "
            "'GPU.0_FP16', 'GPU.1_FP16' or from"
            " HETERO/MULTI/AUTO options available. \n");
      }
    }
    if (provider_options_map.find("device_id") != provider_options_map.end()) {
      device_id = provider_options_map.at("device_id").c_str();
    }
    if (provider_options_map.find("cache_dir") != provider_options_map.end()) {
      cache_dir = provider_options_map.at("cache_dir").c_str();
    }

    if (provider_options_map.find("context") != provider_options_map.end()) {
      std::string str = provider_options_map.at("context");
      unsigned int64_t number = std::strtoull(str.c_str(), nullptr, 16);
      context = reinterpret_cast<void*>(number);
    }

    if (provider_options_map.find("num_of_threads") != provider_options_map.end()) {
      num_of_threads = std::stoi(provider_options_map.at("num_of_threads"));
      if (num_of_threads <= 0) {
        num_of_threads = 1;
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_threads' should be in the positive range.\n "
                              << "Executing with num_threads=1";
      }
    }

    if (provider_options_map.find("num_streams") != provider_options_map.end()) {
      num_streams = std::stoi(provider_options_map.at("num_streams"));
      if (num_streams <= 0) {
        num_streams = 1;
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_streams' should be in the range of 1-8.\n "
                              << "Executing with num_streams=1";
      }
    }
    std::string bool_flag = "";
    if (provider_options_map.find("enable_npu_fast_compile") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("enable_npu_fast_compile");
      if (bool_flag == "true" || bool_flag == "True")
        enable_npu_fast_compile = true;
      else if (bool_flag == "false" || bool_flag == "False")
        enable_npu_fast_compile = false;
      bool_flag = "";
    }

    if (provider_options_map.find("enable_opencl_throttling") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("enable_opencl_throttling");
      if (bool_flag == "true" || bool_flag == "True")
        enable_opencl_throttling = true;
      else if (bool_flag == "false" || bool_flag == "False")
        enable_opencl_throttling = false;
      bool_flag = "";
    }

    if (provider_options_map.find("enable_dynamic_shapes") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("enable_dynamic_shapes");
      if (bool_flag == "true" || bool_flag == "True")
        enable_dynamic_shapes = true;
      else if (bool_flag == "false" || bool_flag == "False")
        enable_dynamic_shapes = false;
    }
    return std::make_shared<OpenVINOProviderFactory>(const_cast<char*>(device_type.c_str()),
                                                     enable_npu_fast_compile,
                                                     device_id,
                                                     num_of_threads,
                                                     cache_dir,
                                                     num_streams,
                                                     context,
                                                     enable_opencl_throttling,
                                                     enable_dynamic_shapes);
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
