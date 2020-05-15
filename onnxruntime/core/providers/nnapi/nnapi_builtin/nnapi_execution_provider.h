// Copyright 2019 JD.com Inc. JD AI

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
class NnapiExecutionProvider : public IExecutionProvider {
 public:
  NnapiExecutionProvider();
  virtual ~NnapiExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;
  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

};
}  // namespace onnxruntime
