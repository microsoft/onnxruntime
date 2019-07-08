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

  for (auto index : order) {
    auto* node = graph.GetNode(index);
    // check that node hasn't already been removed
    if (!node)
      continue;

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

    // we created the new node with the output def from act_node so we don't need to move the definition from act_node
    const bool move_definition = false;
    graph_utils::MoveOutput(graph, act_node, fused_conv, move_definition);

    graph.RemoveNode(conv_node.Index());
    graph.RemoveNode(act_node.Index());

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
