// Copyright 2020 rock-chips.com Inc.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

class RknpuExecutionProvider : public IExecutionProvider {
 public:
  RknpuExecutionProvider();
  virtual ~RknpuExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;
  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  std::unordered_map<std::string, ONNX_NAMESPACE::ModelProto> model_proto_;
  std::unordered_map<std::string, std::unordered_map<std::string, int>> input_info_;
  std::unordered_map<std::string, std::unordered_map<std::string, int>> output_info_;
  std::vector<std::vector<int>> GetSupportedNodes(
      const ONNX_NAMESPACE::ModelProto& model_proto) const;
};
}  // namespace onnxruntime
