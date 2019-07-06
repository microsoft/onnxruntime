// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/conv_activation_fusion.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {
static bool IsFusableActivation(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "LeakyRelu", {6}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sigmoid", {6}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Tanh", {6});
}

}  // namespace

Status ConvActivationFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  std::deque<onnxruntime::NodeIndex> removed_nodes;
  for (auto index : order) {
    auto node = graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Conv", {1}) ||
        !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
        node->GetOutputEdgesCount() != 1) {
      continue;
    }

    const Node& next_node = *(node->OutputNodesBegin());
    if (!IsFusableActivation(next_node) ||
        next_node.GetExecutionProviderType() != node->GetExecutionProviderType()) {
      continue;
    }

    if (graph.IsNodeOutputsInGraphOutputs(*node)) {
      continue;
    }

    Node& conv_node = *node;
    Node& act_node = *graph.GetNode(next_node.Index());

    Node& fused_conv = graph.AddNode(graph.GenerateNodeName("fused " + conv_node.Name()), "FusedConv",
                                     "fused Conv " + conv_node.Name() + "with activation " + act_node.OpType(),
                                     conv_node.MutableInputDefs(),
                                     act_node.MutableOutputDefs(),
                                     &conv_node.GetAttributes(),
                                     "com.microsoft");

    //Add a new attribute to specify the activation type
    fused_conv.AddAttribute("activation", act_node.OpType());

    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_conv.SetExecutionProviderType(conv_node.GetExecutionProviderType());

    //Add optional attributes for activations
    if (act_node.OpType() == "LeakyRelu") {
      const NodeAttributes& attrs = act_node.GetAttributes();
      for (const auto& attr : attrs) {
        fused_conv.AddAttribute(attr.first, attr.second);
      }
    }

    // move edges
    const bool move_definition = false;  // we created the new node with the output def from act_node
    graph_utils::DisconnectNodes(graph, conv_node, act_node, 0);
    graph_utils::MoveOutput(graph, act_node, fused_conv, move_definition);

    removed_nodes.push_front(conv_node.Index());
    removed_nodes.push_front(act_node.Index());
  }

  for (auto node : removed_nodes) {
    // we can directly remove the nodes as
    // a) we checked nothing else depended on the Conv node; and
    // b) we are creating output with the same name as the activation name output and already moved the edges
    graph.RemoveNode(node);
  }

  if (!removed_nodes.empty()) {
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
