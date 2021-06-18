// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <set>
#include "core/framework/execution_provider.h"

namespace onnxruntime {
class InternalTestingExecutionProvider : public IExecutionProvider {
 public:
  InternalTestingExecutionProvider(const std::unordered_set<std::string>& ops,
                                   const std::unordered_set<std::string>& stop_ops = {},
                                   bool debug_output = false);
  virtual ~InternalTestingExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  FusionStyle GetFusionStyle() const override {
    return FusionStyle::FilteredGraphViewer;
  }

 private:
  const std::string ep_name_;

  // List of operators that the EP will claim nodes for
  const std::unordered_set<std::string> ops_;

  // operators that we stop processing at.
  // all nodes of an operator in this list and all their downstream nodes will be skipped
  // e.g. NonMaxSuppression is the beginning of post-processing in an SSD model. It's unsupported for NNAPI,
  //      so from the NMS node on we want to use the CPU EP as the remaining work to do is far cheaper than
  //      the cost of going back to NNAPI.
  const std::unordered_set<std::string> stop_ops_;

  const bool debug_output_;
};
}  // namespace onnxruntime
