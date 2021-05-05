// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "backend_manager.h"
#include <map>
#include <algorithm>
#include <iostream>

namespace onnxruntime {

static void print_build_options() {
  std::cout << "[ERROR] INVALID DEVICE BUILD TYPE SPECIFIED" << std::endl;
  std::cout << "Specify the keyword HETERO (or) MULTI followed by the devices in the order of priority you want to build" << std::endl;
  std::cout << "The different hardware devices that can be added with HETERO/MULTI build ";
  std::cout << "are ['CPU','GPU','MYRIAD','FPGA','HDDL']" << std::endl;
  std::cout << "An example of how to specify the HETERO or MULTI build type. Ex: HETERO:GPU,CPU  Ex: MULTI:MYRIAD,CPU" << std::endl;
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
  std::vector<std::string> dev_options = {"CPU", "GPU", "MYRIAD", "FPGA", "HDDL"};
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
  std::string device_type_;
  std::string precision_;
  bool enable_vpu_fast_compile_;
  std::string device_id_;
  size_t num_of_threads_;
  bool use_compiled_network_;
  std::string blob_dump_path_;
  bool disable_graph_partition_;

  explicit OpenVINOExecutionProviderInfo(std::string dev_type, bool enable_vpu_fast_compile, std::string dev_id, size_t num_of_threads, bool use_compiled_network, std::string blob_dump_path, bool disable_graph_partition)
      : enable_vpu_fast_compile_(enable_vpu_fast_compile), device_id_(dev_id), num_of_threads_(num_of_threads), use_compiled_network_(use_compiled_network), blob_dump_path_(blob_dump_path), disable_graph_partition_(disable_graph_partition) {
    if (dev_type == "") {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP]"
                         << "No runtime device selection option provided.";
#if defined OPENVINO_CONFIG_CPU_FP32
      device_type_ = "CPU";
      precision_ = "FP32";
#elif defined OPENVINO_CONFIG_GPU_FP32
      device_type_ = "GPU";
      precision_ = "FP32";
#elif defined OPENVINO_CONFIG_GPU_FP16
      device_type_ = "GPU";
      precision_ = "FP16";
#elif defined OPENVINO_CONFIG_MYRIAD
      device_type_ = "MYRIAD";
      precision_ = "FP16";
#elif defined OPENVINO_CONFIG_VAD_M
      device_type_ = "HDDL";
      precision_ = "FP16";
#elif defined OPENVINO_CONFIG_VAD_F
      device_type_ = "HETERO:FPGA,CPU";
      precision_ = "FP32";
#elif defined OPENVINO_CONFIG_HETERO || defined OPENVINO_CONFIG_MULTI
#ifdef DEVICE_NAME
#define DEVICE DEVICE_NAME
#endif
      dev_type = DEVICE;
      if (dev_type.find("HETERO") == 0 || dev_type.find("MULTI") == 0) {
        std::vector<std::string> devices = parseDevices(dev_type);
        precision_ = "FP16";
        if (devices[0] == "CPU" || devices[0] == "GPU") {
          precision_ = "FP32";
        }
        device_type_ = dev_type;
      }
#endif
    } else if (dev_type == "CPU_FP32") {
      device_type_ = "CPU";
      precision_ = "FP32";
    } else if (dev_type == "GPU_FP32") {
      device_type_ = "GPU";
      precision_ = "FP32";
    } else if (dev_type == "GPU_FP16") {
      device_type_ = "GPU";
      precision_ = "FP16";
    } else if (dev_type == "MYRIAD_FP16") {
      device_type_ = "MYRIAD";
      precision_ = "FP16";
    } else if (dev_type == "VAD-M_FP16") {
      device_type_ = "HDDL";
      precision_ = "FP16";
    } else if (dev_type == "VAD-F_FP32") {
      device_type_ = "HETERO:FPGA,CPU";
      precision_ = "FP32";
    } else if (dev_type.find("HETERO") == 0 || dev_type.find("MULTI") == 0) {
      std::vector<std::string> devices = parseDevices(dev_type);
      precision_ = "FP16";
      if (devices[0] == "CPU" || devices[0] == "GPU") {
        precision_ = "FP32";
      }
      device_type_ = dev_type;
    } else {
      ORT_THROW("Invalid device string: " + dev_type);
    }
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP]"
                       << "Choosing Device: " << device_type_ << " , Precision: " << precision_;
  }
  OpenVINOExecutionProviderInfo() {
    OpenVINOExecutionProviderInfo("", false, "", 0, false, "", false);
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
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }
};

}  // namespace onnxruntime
