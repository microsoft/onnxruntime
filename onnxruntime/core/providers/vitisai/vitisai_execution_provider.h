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
  std::string backend_type_; 
  int device_id_;
};

}  // namespace onnxruntime
