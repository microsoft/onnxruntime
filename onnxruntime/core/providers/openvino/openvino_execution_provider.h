// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <map>
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <utility>

#include "core/providers/openvino/backend_manager.h"

namespace onnxruntime {

struct OVDevices {
  ov::Core core;
  std::vector<std::string> get_ov_devices() const {
    return core.get_available_devices();
  }
};

static void print_build_options() {
  std::cout << "[ERROR] INVALID DEVICE BUILD TYPE SPECIFIED" << std::endl;
  std::cout << "Specify the keyword HETERO (or) MULTI (or) AUTO followed by the devices in the order of priority "
            << "you want to build"
            << std::endl;
  std::cout << "The different hardware devices that can be added with HETERO/MULTI/AUTO build "
            << "are ['CPU','GPU','NPU','GPU.x'] where x = 0,1,2 and so on"
            << std::endl;
  std::cout << "An example of how to specify the HETERO or MULTI or AUTO build type. "
            << "Ex: HETERO:GPU,CPU  Ex: MULTI:GPU,CPU Ex: AUTO:GPU,CPU Ex: AUTO:GPU.0,CPU Ex: AUTO:GPU.1,CPU"
            << std::endl;
}

static std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    result.push_back(item);
  }
  return result;
}

static std::vector<std::string> parseDevices(const std::string& device_string,
                                             const std::vector<std::string>& available_devices) {
  std::string comma_separated_devices = device_string;
  if (comma_separated_devices.find(":") != std::string::npos) {
    comma_separated_devices = comma_separated_devices.substr(comma_separated_devices.find(":") + 1);
  }
  auto devices = split(comma_separated_devices, ',');
  if (devices.size() < 2) {
    print_build_options();
    ORT_THROW("Invalid device string: " + device_string);
  }
  std::set<std::string> dev_options = {"CPU", "GPU", "NPU"};

  for (auto& device : available_devices) {
    if (dev_options.find(device) == dev_options.end()) {
      auto dev_options_update = dev_options.emplace(device);
    }
  }

  for (const std::string& dev : devices) {
    if (!std::count(dev_options.begin(), dev_options.end(), dev)) {
      print_build_options();
      ORT_THROW("Invalid device string: " + device_string);
    }
  }
  return devices;
}

// Information needed to construct OpenVINO execution providers.
struct OpenVINOExecutionProviderInfo {
  std::string device_type_{""};
  std::string precision_{""};
  bool enable_npu_fast_compile_{false};
  size_t num_of_threads_{0};
  std::string cache_dir_{""};
  std::string model_priority_{""};
  int num_streams_{1};
  void* context_{NULL};
  bool enable_opencl_throttling_{false};
  bool disable_dynamic_shapes_{false};
  bool export_ep_ctx_blob_{false};
  bool enable_qdq_optimizer_{false};
  bool disable_cpu_fallback_{false};
  bool so_epctx_embed_mode_{true};

  OpenVINOExecutionProviderInfo() = delete;

  explicit OpenVINOExecutionProviderInfo(const std::string& dev_type, const std::string& precision,
                                         bool enable_npu_fast_compile, size_t num_of_threads,
                                         const std::string& cache_dir, const std::string& model_priority,
                                         int num_streams, void* context, bool enable_opencl_throttling,
                                         bool disable_dynamic_shapes, bool export_ep_ctx_blob,
                                         bool enable_qdq_optimizer, bool disable_cpu_fallback,
                                         bool so_epctx_embed_mode)
      : precision_(std::move(precision)),
        enable_npu_fast_compile_(enable_npu_fast_compile),
        num_of_threads_(num_of_threads),
        cache_dir_(std::move(cache_dir)),
        model_priority_(std::move(model_priority)),
        num_streams_(num_streams),
        context_(context),
        enable_opencl_throttling_(enable_opencl_throttling),
        disable_dynamic_shapes_(disable_dynamic_shapes),
        export_ep_ctx_blob_(export_ep_ctx_blob),
        enable_qdq_optimizer_(enable_qdq_optimizer),
        disable_cpu_fallback_(disable_cpu_fallback),
        so_epctx_embed_mode_{so_epctx_embed_mode} {
    std::set<std::string> ov_supported_device_types = {"CPU", "GPU",
                                                       "GPU.0", "GPU.1", "NPU"};

    OVDevices devices;
    std::vector<std::string> available_devices = devices.get_ov_devices();

    for (auto& device : available_devices) {
      if (ov_supported_device_types.find(device) == ov_supported_device_types.end()) {
        ov_supported_device_types.emplace(device);
      }
    }

    if (dev_type == "") {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP]"
                         << "No runtime device selection option provided.";
#if defined OPENVINO_CONFIG_CPU
      device_type_ = "CPU";
      precision_ = "FP32";
#elif defined OPENVINO_CONFIG_GPU
      device_type_ = "GPU";
      precision_ = "FP16";
#elif defined OPENVINO_CONFIG_NPU
      device_type_ = "NPU";
      precision_ = "FP16";
#elif defined OPENVINO_CONFIG_HETERO || defined OPENVINO_CONFIG_MULTI || defined OPENVINO_CONFIG_AUTO
#ifdef DEVICE_NAME
#define DEVICE DEVICE_NAME
#endif
      dev_type = DEVICE;

      if (dev_type.find("HETERO") == 0 || dev_type.find("MULTI") == 0 || dev_type.find("AUTO") == 0) {
        std::vector<std::string> devices = parseDevices(dev_type, available_devices);
        precision_ = "FP16";
        if (devices[0] == "CPU") {
          precision_ = "FP32";
        }
        device_type_ = std::move(dev_type);
      }
#endif
    } else if (ov_supported_device_types.find(dev_type) != ov_supported_device_types.end()) {
      device_type_ = std::move(dev_type);
    } else if (dev_type.find("HETERO") == 0 || dev_type.find("MULTI") == 0 || dev_type.find("AUTO") == 0) {
      std::vector<std::string> devices = parseDevices(dev_type, available_devices);
      device_type_ = dev_type;
    } else {
      ORT_THROW("Invalid device string: " + dev_type);
    }
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP]"
                       << "Choosing Device: " << device_type_ << " , Precision: " << precision_;
  }
};

struct OpenVINOEPFunctionState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc destroy_func = nullptr;
  AllocatorHandle allocator_handle = nullptr;
  std::shared_ptr<openvino_ep::BackendManager> backend_manager;
};

// Logical device representation.
class OpenVINOExecutionProvider : public IExecutionProvider {
 public:
  explicit OpenVINOExecutionProvider(const OpenVINOExecutionProviderInfo& info);
  ~OpenVINOExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/) const override;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }
#ifdef USE_DEVICE_MEMORY
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;
#endif
 private:
  std::unique_ptr<openvino_ep::GlobalContext> global_context_;
  openvino_ep::EPCtxHandler ep_ctx_handle_{};
};

}  // namespace onnxruntime
