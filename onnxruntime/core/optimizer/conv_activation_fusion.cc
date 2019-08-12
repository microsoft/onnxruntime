// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/conv_activation_fusion.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {
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

    const auto& next_node = *(node->OutputNodesBegin());

    if (next_node.GetExecutionProviderType() != node->GetExecutionProviderType()) {
      continue;
    }

    if (!graph.GetNodeOutputsInGraphOutputs(*node).empty()) {
      continue;
    }

    // Test if this is an activation that can be fused and also extract the
    // activation's parameters.
    std::vector<float> activation_params;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Relu", {6}) &&
        !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Sigmoid", {6}) &&
        !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Tanh", {6})) {
      if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "LeakyRelu", {6})) {
        activation_params.push_back(graph_utils::GetNodeAttribute(next_node, "alpha")->f());
      } else if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Clip", {6})) {
        activation_params.push_back(graph_utils::GetNodeAttribute(next_node, "min")->f());
        activation_params.push_back(graph_utils::GetNodeAttribute(next_node, "max")->f());
      } else {
        continue;
      }
    }

    Node& conv_node = *node;
    Node& act_node = *graph.GetNode(next_node.Index());

    Node& fused_conv = graph.AddNode(graph.GenerateNodeName("fused " + conv_node.Name()), "FusedConv",
                                     "fused Conv " + conv_node.Name() + "with activation " + act_node.OpType(),
                                     conv_node.MutableInputDefs(),
                                     {},
                                     &conv_node.GetAttributes(),
                                     "com.microsoft");

    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_conv.SetExecutionProviderType(conv_node.GetExecutionProviderType());

    // Add attributes to specify the activation type and parameters.
    fused_conv.AddAttribute("activation", next_node.OpType());
    if (activation_params.size() > 0) {
      fused_conv.AddAttribute("activation_params", activation_params);
    }

    // move output definitions and edges from act_node to fused_conv. delete conv_node and act_node.
    graph_utils::FinalizeNodeFusion(graph, conv_node, act_node, &fused_conv);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
