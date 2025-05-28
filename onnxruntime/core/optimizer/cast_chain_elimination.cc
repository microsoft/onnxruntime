// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/optimizer/cast_chain_elimination.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

static Node* FollowCastChain(Node* current) {
  while (true) {
    // A rare case when the Cast node output is branching out.
    // We don't really want to deal with this complexity, hence we will skip it.
    if (current->GetOutputEdgesCount() > 1) {
      return nullptr;
    }

    auto it = current->OutputNodesBegin();
    if (it == current->OutputNodesEnd()) {
      break;
    }

    Node* next = const_cast<Node*>(&(*it));
    if (next->OpType() != "Cast") {
      break;
    }
    current = next;
  }

  return current;
}

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

    // If not, find the longest chain that casts to the input type, if it exists.
    Node* current = &node;
    Node* final_non_cast_node = FollowCastChain(current);

    if (!final_non_cast_node) {
      continue;
    }

    // No extra casts were found.
    if (node.Index() == final_non_cast_node->Index()) {
      continue;
    }

    std::vector<Node*> to_remove;
    current = &node;

    // Collect nodes for removal.
    while (current != final_non_cast_node && current->OpType() == "Cast") {
      to_remove.push_back(current);
      auto it = current->OutputNodesBegin();
      if (it == current->OutputNodesEnd())
        break;
      current = const_cast<Node*>(&*it);
    }

    // First remove all outbound edges.
    for (Node* n : to_remove) {
      graph_utils::RemoveNodeOutputEdges(graph, *n);
    }

    NodeArg* last_node_output_def = to_remove.back()->MutableOutputDefs()[0];
    const std::string& last_node_output_tensor_name = last_node_output_def->Name();

    // Find the matching def slot, so we can wire the final node to the input of the first removeable node.
    int slot = -1;
    auto& inputs = final_non_cast_node->MutableInputDefs();
    for (int i = 0, n = static_cast<int>(inputs.size()); i < n; ++i) {
      if (inputs[i]->Name() == last_node_output_tensor_name) {
        slot = i;
        break;
      }
    }

    final_non_cast_node->MutableInputDefs()[slot] = to_remove[0]->MutableInputDefs()[0];

    graph_utils::MoveAllNodeInputEdges(graph, *to_remove[0], *final_non_cast_node);

    // Finally, remove the nodes itself.
    for (Node* n : to_remove) {
      graph.RemoveNode(n->Index());
    }

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
