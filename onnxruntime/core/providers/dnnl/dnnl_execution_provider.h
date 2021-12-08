// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <map>
#include <list>
#include <memory.h>

#include "core/platform/ort_mutex.h"
#include "dnnl_op_manager.h"
#include "core/providers/dnnl/subgraph/dnnl_subgraph.h"
#include "core/providers/dnnl/subgraph/dnnl_subgraph_primitive.h"

namespace onnxruntime {

// Information needed to construct DNNL execution providers.
struct DNNLExecutionProviderInfo {
  bool create_arena{true};

  explicit DNNLExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}
  DNNLExecutionProviderInfo() = default;
};

// Logical device representation.
class DNNLExecutionProvider : public IExecutionProvider {
 public:
  explicit DNNLExecutionProvider(const DNNLExecutionProviderInfo& info);
  virtual ~DNNLExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
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
