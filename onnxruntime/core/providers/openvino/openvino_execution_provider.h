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

#include "core/providers/openvino/backend_manager.h"

namespace onnxruntime {

static void print_build_options() {
  std::cout << "[ERROR] INVALID DEVICE BUILD TYPE SPECIFIED" << std::endl;
  std::cout << "Specify the keyword HETERO (or) MULTI (or) AUTO followed by the devices in the order of priority "
            << "you want to build"
            << std::endl;
  std::cout << "The different hardware devices that can be added with HETERO/MULTI/AUTO build "
            << "are ['CPU','GPU','NPU']"
            << std::endl;
  std::cout << "An example of how to specify the HETERO or MULTI or AUTO build type. "
            << "Ex: HETERO:GPU,CPU  Ex: MULTI:GPU,CPU Ex: AUTO:GPU,CPU"
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

static std::vector<std::string> parseDevices(const std::string& device_string) {
  std::string comma_separated_devices = device_string;
  if (comma_separated_devices.find(":") != std::string::npos) {
    comma_separated_devices = comma_separated_devices.substr(comma_separated_devices.find(":") + 1);
  }
  auto devices = split(comma_separated_devices, ',');
  if (devices.size() < 2) {
    print_build_options();
    ORT_THROW("Invalid device string: " + device_string);
  }
  std::vector<std::string> dev_options = {"CPU", "GPU", "NPU"};
  for (std::string dev : devices) {
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

  OpenVINOExecutionProviderInfo() = delete;

  explicit OpenVINOExecutionProviderInfo(std::string dev_type, std::string precision, bool enable_npu_fast_compile,
                                         size_t num_of_threads, std::string cache_dir, std::string model_priority,
                                         int num_streams, void* context, bool enable_opencl_throttling,
                                         bool disable_dynamic_shapes, bool export_ep_ctx_blob)
      : precision_(precision),
        enable_npu_fast_compile_(enable_npu_fast_compile),
        num_of_threads_(num_of_threads),
        cache_dir_(cache_dir),
        model_priority_(model_priority),
        num_streams_(num_streams),
        context_(context),
        enable_opencl_throttling_(enable_opencl_throttling),
        disable_dynamic_shapes_(disable_dynamic_shapes),
        export_ep_ctx_blob_(export_ep_ctx_blob) {
    std::set<std::string> ov_supported_device_types = {"CPU", "GPU",
                                                       "GPU.0", "GPU.1", "NPU"};
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
        std::vector<std::string> devices = parseDevices(dev_type);
        precision_ = "FP16";
        if (devices[0] == "CPU") {
          precision_ = "FP32";
        }
        device_type_ = dev_type;
      }
#endif
    } else if (ov_supported_device_types.find(dev_type) != ov_supported_device_types.end()) {
      device_type_ = dev_type;
    } else if (dev_type.find("HETERO") == 0 || dev_type.find("MULTI") == 0 || dev_type.find("AUTO") == 0) {
      std::vector<std::string> devices = parseDevices(dev_type);
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

 private:
  std::unique_ptr<openvino_ep::GlobalContext> global_context_;
  openvino_ep::EPCtxHandler ep_ctx_handle_{};
};

}  // namespace onnxruntime
