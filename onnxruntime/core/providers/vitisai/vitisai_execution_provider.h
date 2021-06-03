// Copyright (c) Xilinx Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ctime>
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

// Information needed to construct execution providers.
struct VitisAIExecutionProviderInfo {
  int device_id{0};
  std::string backend_type;
  std::string export_runtime_module;
  std::string load_runtime_module;
};

// Logical device representation.
class VitisAIExecutionProvider : public IExecutionProvider {
 public:
  explicit VitisAIExecutionProvider(const VitisAIExecutionProviderInfo& info);
  ~VitisAIExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  int GetDeviceId() const { return device_id_; }

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  // The Vitis AI DPU target
  std::string backend_type_;
  // Device ID (Unused for now)
  int device_id_;
  // If not empty, the path to the file where the PyXIR runtime module
  //	should be exported to (used for cross compilation)
  std::string export_runtime_module_;
  // If not empty, the path to the file where the PyXIR runtime module
  //	should be loaded from
  std::string load_runtime_module_;
};

}  // namespace onnxruntime
