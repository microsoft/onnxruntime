// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "backend_manager.h"
#include <map>
#include <algorithm>
#include <iostream>
#include "interface/provider/provider.h"

namespace onnxruntime {

static void print_build_options() {
  std::cout << "[ERROR] INVALID DEVICE BUILD TYPE SPECIFIED" << std::endl;
  std::cout << "Specify the keyword HETERO (or) MULTI (or) AUTO followed by the devices in the order of priority you want to build" << std::endl;
  std::cout << "The different hardware devices that can be added with HETERO/MULTI/AUTO build ";
  std::cout << "are ['CPU','GPU','VPUX']" << std::endl;
  std::cout << "An example of how to specify the HETERO or MULTI or AUTO build type. Ex: HETERO:GPU,CPU  Ex: MULTI:GPU,CPU Ex: AUTO:GPU,CPU" << std::endl;
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
  std::vector<std::string> dev_options = {"CPU", "GPU", "VPUX"};
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
  std::string cache_dir_;
  int num_streams_;
  void* context_;
  bool enable_opencl_throttling_;
  bool enable_dynamic_shapes_;

  explicit OpenVINOExecutionProviderInfo(std::string dev_type, bool enable_vpu_fast_compile, std::string dev_id,
                                         size_t num_of_threads, std::string cache_dir, int num_streams,
                                         void* context, bool enable_opencl_throttling,
                                         bool enable_dynamic_shapes)
      : enable_vpu_fast_compile_(enable_vpu_fast_compile), device_id_(dev_id), num_of_threads_(num_of_threads), cache_dir_(cache_dir), num_streams_(num_streams), context_(context), enable_opencl_throttling_(enable_opencl_throttling), enable_dynamic_shapes_(enable_dynamic_shapes) {
    if (dev_type == "") {
      //LOGS_DEFAULT(INFO) << "[OpenVINO-EP]"
      //                   << "No runtime device selection option provided.";
#if defined OPENVINO_CONFIG_CPU_FP32
      device_type_ = "CPU";
      precision_ = "FP32";
#elif defined OPENVINO_CONFIG_CPU_FP16
      device_type_ = "CPU";
      precision_ = "FP16";
#elif defined OPENVINO_CONFIG_GPU_FP32
      device_type_ = "GPU";
      precision_ = "FP32";
#elif defined OPENVINO_CONFIG_GPU_FP16
      device_type_ = "GPU";
      precision_ = "FP16";
#elif defined OPENVINO_CONFIG_VPUX_FP16
      device_type_ = "VPUX";
      precision_ = "FP16";
#elif defined OPENVINO_CONFIG_VPUX_U8
      device_type_ = "VPUX";
      precision_ = "U8";
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
    } else if (dev_type == "CPU_FP32") {
      device_type_ = "CPU";
      precision_ = "FP32";
    } else if (dev_type == "CPU_FP16") {
      device_type_ = "CPU";
      precision_ = "FP16";
    } else if (dev_type == "GPU_FP32") {
      device_type_ = "GPU";
      precision_ = "FP32";
    } else if (dev_type == "GPU.0_FP32") {
      device_type_ = "GPU.0";
      precision_ = "FP32";
    } else if (dev_type == "GPU.1_FP32") {
      device_type_ = "GPU.1";
      precision_ = "FP32";
    } else if (dev_type == "GPU_FP16") {
      device_type_ = "GPU";
      precision_ = "FP16";
    } else if (dev_type == "GPU.0_FP16") {
      device_type_ = "GPU.0";
      precision_ = "FP16";
    } else if (dev_type == "GPU.1_FP16") {
      device_type_ = "GPU.1";
      precision_ = "FP16";
    } else if (dev_type == "VPUX_FP16") {
      device_type_ = "VPUX";
      precision_ = "FP16";
    } else if (dev_type == "VPUX_U8") {
      device_type_ = "VPUX";
      precision_ = "U8";
    } else if (dev_type.find("HETERO") == 0 || dev_type.find("MULTI") == 0) {
      std::vector<std::string> devices = parseDevices(dev_type);
      precision_ = "FP16";
      if (devices[0] == "CPU") {
        precision_ = "FP32";
      }
      device_type_ = dev_type;
    } else if (dev_type.find("AUTO") == 0) {
      std::vector<std::string> devices = parseDevices(dev_type);
      precision_ = "FP32";
      device_type_ = dev_type;
    } else {
      ORT_THROW("Invalid device string: " + dev_type);
    }
    //LOGS_DEFAULT(INFO) << "[OpenVINO-EP]"
    //                   << "Choosing Device: " << device_type_ << " , Precision: " << precision_;
  }
  OpenVINOExecutionProviderInfo() {
    OpenVINOExecutionProviderInfo("", false, "", 0, "", 1, NULL, false, false);
  }
};

struct OpenVINOEPFunctionState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc destroy_func = nullptr;
  AllocatorHandle allocator_handle = nullptr;
  std::shared_ptr<openvino_ep::BackendManager> backend_manager;
};

// Logical device representation.
class OpenVINOExecutionProvider : public interface::ExecutionProvider {
 public:
  explicit OpenVINOExecutionProvider(const OpenVINOExecutionProviderInfo& info);
  ~OpenVINOExecutionProvider() = default;

  std::vector<std::unique_ptr<interface::SubGraphDef>> GetCapability(interface::GraphViewRef*) override;

  Status Compile(std::vector<std::unique_ptr<interface::GraphViewRef>>&, std::vector<std::unique_ptr<interface::NodeViewRef>>&, std::vector<NodeComputeInfo>&) override;

  void RegisterKernels(interface::IKernelRegistry&) override {}
};

}  // namespace onnxruntime
