// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

class ShlExecutionProvider : public IExecutionProvider {
 public:
  explicit ShlExecutionProvider(const std::unordered_map<std::string, std::string>& config);
  ~ShlExecutionProvider() = default;
  const std::unordered_map<std::string, std::string>& GetConfig() const { return config_; }
  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const IKernelLookup& /*kernel_lookup*/) const override;
  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  std::vector<std::vector<int>> GetSupportedNodes(const onnxruntime::GraphViewer& graph_viewer) const;

  std::unordered_map<std::string, std::string> config_;
};

}  // namespace onnxruntime
