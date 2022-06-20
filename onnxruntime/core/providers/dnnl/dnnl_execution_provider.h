// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <list>
#include <memory>
#include <memory.h>

#include "core/providers/dnnl/dnnl_op_manager.h"
#include "core/providers/dnnl/dnnl_custom_threadpool.h"
#include "core/providers/dnnl/subgraph/dnnl_subgraph.h"
#include "core/providers/dnnl/subgraph/dnnl_subgraph_primitive.h"

namespace onnxruntime {

// Information needed to construct DNNL execution providers.
struct DNNLExecutionProviderInfo {
  bool create_arena{true};
  void* threadpool_args{nullptr};

  explicit DNNLExecutionProviderInfo(bool use_arena, void* threadpool_args)
      : create_arena(use_arena),
        threadpool_args(threadpool_args) {}
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

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  common::Status SetComputeStream(void* threadpool_args) override;
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
  // This threadpool can be defined at EP creation time, by using
  // DNNLExecutionProviderInfo.ort_threadpool when using a standalone threadpool other than OpenMP
  // or at InferenceSession creation time when using ORT's threadpool (this is done automagically),
  // if a valid DNNLExecutionProviderInfo.ort_threadpool is passed then this one is used over the
  // one passed at InferenceSession creation time.
  std::unique_ptr<DnnlThreadPoolIface> threadpool_;
};

}  // namespace onnxruntime
