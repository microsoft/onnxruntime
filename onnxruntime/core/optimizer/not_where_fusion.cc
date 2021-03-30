// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/not_where_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

/**
Transform that fuses two Not -> Where nodes to a single Where node
with the where inputs 1 and 2 flipped.
Condition ->  Not -> Where ->
              value0-|  |
              value1----|

Condition -> Where ->
      value1-|  |
      value0----|
 */
Status NotWhereFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Where", {9})) {
      continue;
    }

    const Node* p_not_node = graph_utils::GetInputNode(node, 0);
    if (p_not_node == nullptr ||
        !graph_utils::IsSupportedOptypeVersionAndDomain(*p_not_node, "Not", {1}) ||
        p_not_node->GetOutputEdgesCount() != 1 ||
        // Make sure the two nodes do not span execution providers.
        p_not_node->GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    if (!graph.GetNodeOutputsInGraphOutputs(*p_not_node).empty()) {
      continue;
    }

    auto& where_node = node;
    auto& not_node = *graph.GetNode(p_not_node->Index());  // get mutable next node
    NodeArg* not_input = not_node.MutableInputDefs()[0];
    std::vector<NodeArg*> where_inputs = where_node.MutableInputDefs();

    graph_utils::RemoveNodeOutputEdges(graph, not_node);
    graph.RemoveNode(not_node.Index());

    graph_utils::ReplaceNodeInput(where_node, 0, *not_input);
    graph_utils::ReplaceNodeInput(where_node, 1, *where_inputs[2]);
    graph_utils::ReplaceNodeInput(where_node, 2, *where_inputs[1]);
  }

  modified = true;

  return Status::OK();
}
}  // namespace onnxruntime
