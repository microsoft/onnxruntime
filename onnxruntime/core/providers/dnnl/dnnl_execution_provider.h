// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <list>
#include <memory>
#include <memory.h>

#include "core/providers/dnnl/dnnl_execution_provider_info.h"
#include "core/providers/dnnl/dnnl_threadpool.h"
#include "core/providers/dnnl/dnnl_op_manager.h"
#include "core/providers/dnnl/subgraph/dnnl_subgraph.h"
#include "core/providers/dnnl/subgraph/dnnl_subgraph_primitive.h"

namespace onnxruntime {

// Logical device representation.
class DnnlExecutionProvider : public IExecutionProvider {
 public:
  explicit DnnlExecutionProvider(const DnnlExecutionProviderInfo& info);
  virtual ~DnnlExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const IKernelLookup& /*kernel_lookup*/) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  DnnlExecutionProviderInfo info_;
  // DnnlOpManager contains information about supported Dnnl Operators
  DnnlOpManager opManager_;
  std::unordered_map<std::string, std::unique_ptr<ort_dnnl::DnnlSubgraph>> subgraphs_;
  std::unordered_map<std::string, std::unique_ptr<ort_dnnl::DnnlSubgraphPrimitive>> subgraph_primitives_;
  std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer) const;
  // dump subgraphs to onnx format for debugging purpose
  bool dump_subgraphs_ = false;
  bool debug_log_ = false;
  //enable fusion by default
  bool enable_fusion_ = true;
};

}  // namespace onnxruntime
