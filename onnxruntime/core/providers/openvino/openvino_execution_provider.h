// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/allocatormgr.h"
#include "backend_manager.h"
#include <map>

namespace onnxruntime {

// Information needed to construct OpenVINO execution providers.
struct OpenVINOExecutionProviderInfo {
  std::string device_id_;
  std::string precision_;
  bool enable_vpu_fast_compile_;

  explicit OpenVINOExecutionProviderInfo(std::string dev_id, bool enable_vpu_fast_compile = false)
            : enable_vpu_fast_compile_(enable_vpu_fast_compile) {

    if (dev_id == "") {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP]"
                         << "No runtime device selection option provided.";
      #if defined OPENVINO_CONFIG_CPU_FP32
      device_id_ = "CPU";
      precision_ = "FP32";
      #elif defined OPENVINO_CONFIG_GPU_FP32
      device_id_ = "GPU";
      precision_ = "FP32";
      #elif defined OPENVINO_CONFIG_GPU_FP16
      device_id_ = "GPU";
      precision_ = "FP16";
      #elif defined OPENVINO_CONFIG_MYRIAD
      device_id_ = "MYRIAD";
      precision_ = "FP16";
      #elif defined OPENVINO_CONFIG_VAD_M
      device_id_ = "HDDL";
      precision_ = "FP16";
      #elif defined OPENVINO_CONFIG_VAD_F
      device_id_ = "HETERO:FPGA,CPU";
      precision_ = "FP32";
      #endif
    } else if (dev_id == "CPU_FP32") {
      device_id_ = "CPU";
      precision_ = "FP32";
    } else if (dev_id == "GPU_FP32") {
      device_id_ = "GPU";
      precision_ = "FP32";
    } else if (dev_id == "GPU_FP16") {
      device_id_ = "GPU";
      precision_ = "FP16";
    } else if (dev_id == "MYRIAD_FP16") {
      device_id_ = "MYRIAD";
      precision_ = "FP16";
    } else if (dev_id == "VAD-M_FP16") {
      device_id_ = "HDDL";
      precision_ = "FP16";
    } else if (dev_id == "VAD-F_FP32") {
      device_id_ = "HETERO:FPGA,CPU";
      precision_ = "FP32";
    } else {
      ORT_THROW("Invalid device string: " + dev_id);
    }
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP]"
                       << "Choosing Device: " << device_id_ << " , Precision: " << precision_;
  }
  OpenVINOExecutionProviderInfo() {
    OpenVINOExecutionProviderInfo("");
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
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return std::make_shared<KernelRegistry>();
  }

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }
 private:
  OpenVINOExecutionProviderInfo info_;
};

}  // namespace onnxruntime
