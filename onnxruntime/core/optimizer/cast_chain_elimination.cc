// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/optimizer/cast_chain_elimination.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status CastChainElimination::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& node = *node_ptr;

    if (node.OpType() != "Cast") {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (graph_utils::IsGraphOutput(graph, node.OutputDefs()[0])) {
      continue;
    }

    const auto* input_type = node.InputDefs()[0]->TypeAsProto();
    if (input_type == nullptr || !input_type->tensor_type().has_elem_type()) {
      continue;
    }

    if (!graph_utils::CanRemoveNode(graph, node, logger)) {
      continue;
    }

    // Skip nodes that don't have 1 output edge.
    if (node.GetOutputEdgesCount() != 1) {
      continue;
    }

    auto nextNodeIt = node.OutputNodesBegin();

    Node* next = graph.GetNode(nextNodeIt->Index());

    // Skip if the next node is not of type Cast.
    if (next->OpType() != "Cast") {
      continue;
    }

    // We can remove the current node.
    graph_utils::RemoveNodeOutputEdges(graph, node);

    NodeArg* last_node_output_def = node.MutableOutputDefs()[0];
    const std::string& last_node_output_tensor_name = last_node_output_def->Name();

    // Find the matching def slot, so we can wire the final node to the input of the removeable node.
    int slot = -1;
    auto& inputs = next->MutableInputDefs();
    for (int i = 0, n = static_cast<int>(inputs.size()); i < n; ++i) {
      if (inputs[i]->Name() == last_node_output_tensor_name) {
        slot = i;
        break;
      }
    }

    next->MutableInputDefs()[slot] = node.MutableInputDefs()[0];

    graph_utils::MoveAllNodeInputEdges(graph, node, *next);

    graph.RemoveNode(node.Index());

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
