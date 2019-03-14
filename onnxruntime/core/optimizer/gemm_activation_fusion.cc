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
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "LeakyRelu", 6) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", 6) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sigmoid", 6) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Tanh", 6);
}

void HandleActivationNodeEdges(Graph& g, const Node& act, Node& fused_gemm) {
  Node::EdgeSet output_edges;
  for (auto it = act.OutputEdgesBegin(); it != act.OutputEdgesEnd(); ++it) {
    output_edges.insert(*it);
  }

  //remove output edge of activation
  //connect fused_gemm node and nodes after activation nodes
  for (auto& output_edge : output_edges) {
    NodeIndex dst_node_index = output_edge.GetNode().Index();
    int src_arg_index = output_edge.GetSrcArgIndex();
    int dst_arg_index = output_edge.GetDstArgIndex();
    g.RemoveEdge(act.Index(), dst_node_index, src_arg_index, dst_arg_index);
    g.AddEdge(fused_gemm.Index(), dst_node_index, 0, dst_arg_index);
  }
}

}  // namespace

Status GemmActivationFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  std::deque<onnxruntime::NodeIndex> removed_nodes;
  for (auto index : order) {
    auto& node = *graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", 7) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", 9)) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }
    const Node& next_node = *(node.OutputNodesBegin());
    if (!IsFusableActivation(next_node)) {
      continue;
    }

    Node& gemm_node = node;
    const Node& act_node = next_node;

    Node& fused_gemm = graph.AddNode(graph.GenerateNodeName("fused " + gemm_node.Name()), "FusedGemm",
                                     "fused Gemm " + gemm_node.Name() + "with activation " + act_node.OpType(),
                                     gemm_node.MutableInputDefs(),
                                     graph.IsNodeOutputsInGraphOutputs(next_node)
                                         ? const_cast<Node&>(act_node).MutableOutputDefs()
                                         : gemm_node.MutableOutputDefs(),
                                     &gemm_node.GetAttributes(),
                                     "com.microsoft");

    //Add a new attribute to specify the activation type
    fused_gemm.AddAttribute("activation", act_node.OpType());

    //Add optional attributes for activations
    if (act_node.OpType() == "LeakyRelu") {
      const NodeAttributes attrs = act_node.GetAttributes();
      for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        fused_gemm.AddAttribute("leaky_relu_" + it->first, it->second);
      }
    }

    if (!graph.IsNodeOutputsInGraphOutputs(next_node)) {
      HandleActivationNodeEdges(graph, act_node, fused_gemm);

      // Replace the input of the node following activation node
      const NodeArg* act_output_def = act_node.OutputDefs()[0];
      NodeArg* fused_gemm_output_def = fused_gemm.MutableOutputDefs()[0];
      for (auto it = act_node.OutputNodesBegin(); it != act_node.OutputNodesEnd(); ++it) {
        auto output_node = graph.GetNode((*it).Index());
        if (!output_node) {
          return Status(ONNXRUNTIME, INVALID_ARGUMENT);
        }

        auto& input_defs = output_node->MutableInputDefs();
        for (auto& def : input_defs) {
          if (def == act_output_def) {
            def = fused_gemm_output_def;
          }
        }
      }
    }

    removed_nodes.push_front(gemm_node.Index());
    removed_nodes.push_front(act_node.Index());
  }

  for (auto node : removed_nodes) {
    graph.RemoveNode(node);
  }

  if (!removed_nodes.empty()) {
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
