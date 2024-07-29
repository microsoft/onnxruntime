// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"

namespace onnxruntime {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(const char* device_type, const char* precision,
                          bool enable_npu_fast_compile, size_t num_of_threads,
                          const char* cache_dir, const char* model_priority,
                          int num_streams, void* context,
                          bool enable_opencl_throttling, bool disable_dynamic_shapes,
                          bool export_ep_ctx_blob, bool enable_qdq_optimizer,
                          bool disable_cpu_fallback,
                          bool so_epctx_embed_mode)
      : precision_(precision),
        enable_npu_fast_compile_(enable_npu_fast_compile),
        num_of_threads_(num_of_threads),
        model_priority_(model_priority),
        num_streams_(num_streams),
        context_(context),
        enable_opencl_throttling_(enable_opencl_throttling),
        disable_dynamic_shapes_(disable_dynamic_shapes),
        export_ep_ctx_blob_(export_ep_ctx_blob),
        enable_qdq_optimizer_(enable_qdq_optimizer),
        disable_cpu_fallback_(disable_cpu_fallback),
        so_epctx_embed_mode_(so_epctx_embed_mode) {
    device_type_ = (device_type == nullptr) ? "" : device_type;
    cache_dir_ = (cache_dir == nullptr) ? "" : cache_dir;
  }

  ~OpenVINOProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  std::string device_type_;
  std::string precision_;
  bool enable_npu_fast_compile_;
  size_t num_of_threads_;
  std::string cache_dir_;
  std::string model_priority_;
  int num_streams_;
  void* context_;
  bool enable_opencl_throttling_;
  bool disable_dynamic_shapes_;
  bool export_ep_ctx_blob_;
  bool enable_qdq_optimizer_;
  bool disable_cpu_fallback_;
  bool so_epctx_embed_mode_;
};

std::unique_ptr<IExecutionProvider> OpenVINOProviderFactory::CreateProvider() {
  OpenVINOExecutionProviderInfo info(device_type_, precision_, enable_npu_fast_compile_, num_of_threads_,
                                     cache_dir_, model_priority_, num_streams_, context_, enable_opencl_throttling_,
                                     disable_dynamic_shapes_, export_ep_ctx_blob_, enable_qdq_optimizer_,
                                     disable_cpu_fallback_,
                                     so_epctx_embed_mode_);
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

    std::string device_type = "";            // [device_type]: Overrides the accelerator hardware type and precision
                                             //   with these values at runtime.
    std::string precision = "";              // [precision]: Sets the inference precision for execution.
                                             // Supported precision for devices are CPU=FP32, GPU=FP32,FP16, NPU=FP16.
                                             // Not setting precision will execute with optimized precision for
                                             // best inference latency. set Precision=ACCURACY for executing models
                                             // with input precision for best accuracy.
    bool enable_npu_fast_compile = false;    // [enable_npu_fast_compile]: Fast-compile may be optionally enabled to
                                             // speeds up the model's compilation to NPU device specific format.
    int num_of_threads = 0;                  // [num_of_threads]: Overrides the accelerator default value of number of
                                             //  threads with this value at runtime.
    const char* cache_dir = "";              // [cache_dir]: specify the path to
                                             // dump and load the blobs for the model caching/kernel caching (GPU)
                                             // feature. If blob files are already present, it will be directly loaded.
    const char* model_priority = "DEFAULT";  // High-level OpenVINO model priority hint
                                             // Defines what model should be provided with more performant
                                             // bounded resource first
    int num_streams = 1;                     // [num_streams]: Option that specifies the number of parallel inference
                                             // requests to be processed on a given `device_type`. Overrides the
                                             // accelerator default value of number of streams
                                             // with this value at runtime.
    bool enable_opencl_throttling = false;   // [enable_opencl_throttling]: Enables OpenCL queue throttling for GPU
                                             // device (Reduces CPU Utilization when using GPU)
    bool export_ep_ctx_blob = false;         // Whether to export the pre-compiled blob as an EPContext model.

    void* context = nullptr;

    bool enable_qdq_optimizer = false;

    bool disable_cpu_fallback = false;

    bool so_epctx_embed_mode = true;

    if (provider_options_map.find("device_type") != provider_options_map.end()) {
      device_type = provider_options_map.at("device_type").c_str();

      std::set<std::string> ov_supported_device_types = {"CPU", "GPU",
                                                         "GPU.0", "GPU.1", "NPU"};
      std::set<std::string> deprecated_device_types = {"CPU_FP32", "GPU_FP32",
                                                       "GPU.0_FP32", "GPU.1_FP32", "GPU_FP16",
                                                       "GPU.0_FP16", "GPU.1_FP16"};
      OVDevices devices;
      std::vector<std::string> available_devices = devices.get_ov_devices();

      for (auto& device : available_devices) {
        if (ov_supported_device_types.find(device) == ov_supported_device_types.end()) {
          ov_supported_device_types.emplace(device);
        }
      }
      if (deprecated_device_types.find(device_type) != deprecated_device_types.end()) {
        std::string deprecated_device = device_type;
        int delimit = device_type.find("_");
        device_type = deprecated_device.substr(0, delimit);
        precision = deprecated_device.substr(delimit + 1);
        LOGS_DEFAULT(WARNING) << "[OpenVINO] Selected 'device_type' " + deprecated_device + " is deprecated. \n"
                              << "Update the 'device_type' to specified types 'CPU', 'GPU', 'GPU.0', "
                              << "'GPU.1', 'NPU' or from"
                              << " HETERO/MULTI/AUTO options and set 'precision' separately. \n";
      }
      if (!((ov_supported_device_types.find(device_type) != ov_supported_device_types.end()) ||
            (device_type.find("HETERO:") == 0) ||
            (device_type.find("MULTI:") == 0) ||
            (device_type.find("AUTO:") == 0))) {
        ORT_THROW(
            "[ERROR] [OpenVINO] You have selected wrong configuration value for the key 'device_type'. "
            "Select from 'CPU', 'GPU', 'NPU', 'GPU.x' where x = 0,1,2 and so on or from"
            " HETERO/MULTI/AUTO options available. \n");
      }
    }
    if (provider_options_map.find("device_id") != provider_options_map.end()) {
      std::string dev_id = provider_options_map.at("device_id").c_str();
      LOGS_DEFAULT(WARNING) << "[OpenVINO] The options 'device_id' is deprecated. "
                            << "Upgrade to set deice_type and precision session options.\n";
      if (dev_id == "CPU" || dev_id == "GPU" || dev_id == "NPU") {
        device_type = std::move(dev_id);
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported device_id is selected. Select from available options.");
      }
    }
    if (provider_options_map.find("precision") != provider_options_map.end()) {
      precision = provider_options_map.at("precision").c_str();
    }
    if (device_type.find("GPU") != std::string::npos) {
      if (precision == "") {
        precision = "FP16";
      } else if (precision != "ACCURACY" && precision != "FP16" && precision != "FP32") {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. GPU only supports FP32 / FP16. \n");
      }
    } else if (device_type.find("NPU") != std::string::npos) {
      if (precision == "" || precision == "ACCURACY" || precision == "FP16") {
        precision = "FP16";
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. NPU only supported FP16. \n");
      }
    } else if (device_type.find("CPU") != std::string::npos) {
      if (precision == "" || precision == "ACCURACY" || precision == "FP32") {
        precision = "FP32";
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. CPU only supports FP32 . \n");
      }
    }

    if (provider_options_map.find("cache_dir") != provider_options_map.end()) {
      cache_dir = provider_options_map.at("cache_dir").c_str();
    }

    if (provider_options_map.find("context") != provider_options_map.end()) {
      std::string str = provider_options_map.at("context");
      uint64_t number = std::strtoull(str.c_str(), nullptr, 16);
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

    if (provider_options_map.find("model_priority") != provider_options_map.end()) {
      model_priority = provider_options_map.at("model_priority").c_str();
      std::vector<std::string> supported_priorities({"LOW", "MEDIUM", "HIGH", "DEFAULT"});
      if (std::find(supported_priorities.begin(), supported_priorities.end(),
                    model_priority) == supported_priorities.end()) {
        model_priority = "DEFAULT";
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'model_priority' "
                              << "is not one of LOW, MEDIUM, HIGH, DEFAULT. "
                              << "Executing with model_priorty=DEFAULT";
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

    if (provider_options_map.find("enable_qdq_optimizer") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("enable_qdq_optimizer");
      if (bool_flag == "true" || bool_flag == "True")
        enable_qdq_optimizer = true;
      else if (bool_flag == "false" || bool_flag == "False")
        enable_qdq_optimizer = false;
      bool_flag = "";
    }

    // [disable_dynamic_shapes]:  Rewrite dynamic shaped models to static shape at runtime and execute.
    // Always true for NPU plugin.
    bool disable_dynamic_shapes = false;
    if (device_type.find("NPU") != std::string::npos) {
      disable_dynamic_shapes = true;
    }
    if (provider_options_map.find("disable_dynamic_shapes") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("disable_dynamic_shapes");
      if (bool_flag == "true" || bool_flag == "True") {
        disable_dynamic_shapes = true;
      } else if (bool_flag == "false" || bool_flag == "False") {
        if (device_type.find("NPU") != std::string::npos) {
          disable_dynamic_shapes = true;
          LOGS_DEFAULT(INFO) << "[OpenVINO-EP] The value for the key 'disable_dynamic_shapes' will be set to "
                             << "TRUE for NPU backend.\n ";
        } else {
          disable_dynamic_shapes = false;
        }
      }
    }
    if (provider_options_map.find("so_export_ep_ctx_blob") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("so_export_ep_ctx_blob");
      if (bool_flag == "true" || bool_flag == "True")
        export_ep_ctx_blob = true;
      else if (bool_flag == "false" || bool_flag == "False")
        export_ep_ctx_blob = false;
      bool_flag = "";
    }

    if (provider_options_map.find("disable_cpu_fallback") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("disable_cpu_fallback");
      if (bool_flag == "true" || bool_flag == "True")
        disable_cpu_fallback = true;
      else if (bool_flag == "false" || bool_flag == "False")
        disable_cpu_fallback = false;
      bool_flag = "";
    }
    if (provider_options_map.find("so_epctx_embed_mode") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("so_epctx_embed_mode");
      if (bool_flag == "true" || bool_flag == "True")
        so_epctx_embed_mode = true;
      else if (bool_flag == "false" || bool_flag == "False")
        so_epctx_embed_mode = false;
      bool_flag = "";
    }

    if (provider_options_map.find("so_epctx_path") != provider_options_map.end()) {
      // The path to dump epctx model is valid only when epctx is enabled.
      // Overrides the cache_dir option to dump model cache files from OV.
      if (export_ep_ctx_blob) {
        cache_dir = provider_options_map.at("so_epctx_path").c_str();
      }
    }

    return std::make_shared<OpenVINOProviderFactory>(const_cast<char*>(device_type.c_str()),
                                                     const_cast<char*>(precision.c_str()),
                                                     enable_npu_fast_compile,
                                                     num_of_threads,
                                                     cache_dir,
                                                     model_priority,
                                                     num_streams,
                                                     context,
                                                     enable_opencl_throttling,
                                                     disable_dynamic_shapes,
                                                     export_ep_ctx_blob,
                                                     enable_qdq_optimizer,
                                                     disable_cpu_fallback,
                                                     so_epctx_embed_mode);
  }

  void Initialize() override {
  }

  void Shutdown() override {
  }
} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
