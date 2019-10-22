// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {
bool IsFusableActivation(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "LeakyRelu", {6}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sigmoid", {6}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Tanh", {6});
}
}  // namespace

Status GemmActivationFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  std::deque<onnxruntime::NodeIndex> removed_nodes;
  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", {7, 9}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    const Node& next_node = *(node.OutputNodesBegin());
    if (!IsFusableActivation(next_node) ||
        next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    if (!graph.GetNodeOutputsInGraphOutputs(node).empty()) {
      continue;
    }

    Node& gemm_node = node;
    Node& act_node = *graph.GetNode(next_node.Index());  // get mutable reference

    Node& fused_gemm = graph.AddNode(graph.GenerateNodeName("fused " + gemm_node.Name()), "FusedGemm",
                                     "fused Gemm " + gemm_node.Name() + "with activation " + act_node.OpType(),
                                     gemm_node.MutableInputDefs(),
                                     {},
                                     &gemm_node.GetAttributes(),
                                     "com.microsoft");

    //Add a new attribute to specify the activation type
    fused_gemm.AddAttribute("activation", act_node.OpType());

    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_gemm.SetExecutionProviderType(gemm_node.GetExecutionProviderType());

    //Add optional attributes for activations
    if (act_node.OpType() == "LeakyRelu") {
      const NodeAttributes& attrs = act_node.GetAttributes();
      for (const auto& attr : attrs) {
        fused_gemm.AddAttribute("leaky_relu_" + attr.first, attr.second);
      }
    }

    // move output definitions and edges from act_node to fused_gemm. delete gemm_node and act_node.
    graph_utils::FinalizeNodeFusion(graph, {gemm_node, act_node}, fused_gemm);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
