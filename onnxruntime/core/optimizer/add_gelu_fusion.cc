// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/add_gelu_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status AddGeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {7}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    auto next_node_itr = node.OutputNodesBegin();
    if (next_node_itr == node.OutputNodesEnd()) {
      continue;
    }

    const Node& next_node = (*next_node_itr);
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Gelu", {1}, kMSDomain) ||
        next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    if (!graph.GetNodeOutputsInGraphOutputs(node).empty()) {
      continue;
    }

    Node& add_node = node;
    Node& gelu_node = const_cast<Node&>(next_node);

    Node& gelu_add_fusion_node = graph.AddNode(graph.GenerateNodeName("AddGeluFusion"),
                                               "AddGeluFusion",
                                               "fused Add and Gelu",
                                               {add_node.MutableInputDefs()},
                                               {},
                                               {},
                                               kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gelu_add_fusion_node.SetExecutionProviderType(gelu_node.GetExecutionProviderType());

    // move output definitions and edges from gelu_node to gelu_add_fusion_node
    //delete add_node and gelu_node.
    graph_utils::FinalizeNodeFusion(graph, {add_node, gelu_node}, gelu_add_fusion_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
