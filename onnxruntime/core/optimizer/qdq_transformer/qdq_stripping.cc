// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_stripping.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/framework/node_unit.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {
/**
 */
Status RemoveQDQ(Graph& graph, 
                 std::unordered_map<Node*, Node*>& candidate_dq_to_q_map,
                 std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                 bool& modified) {
  // Remove DQ output edges and node args
  for (auto qdq_pair : candidate_dq_to_q_map) {
    auto dq_node = qdq_pair.first;
    auto q_node = qdq_pair.second;
    assert(node_unit_map.find(dq_node) != node_unit_map.end());
    auto node_unit = node_unit_map[dq_node];

    // Get node args for replacement (The dst replacement should be mutable)
    assert(q_node->InputDefs().size() == 3);   // Q node should have three inputs
    assert(dq_node->OutputDefs().size() == 1); // DQ node should have one output
    auto q_node_first_input_arg = q_node->MutableInputDefs()[0];
    auto dq_node_output_arg = dq_node->OutputDefs()[0];

    // Remove DQ output edges
    graph_utils::RemoveNodeOutputEdges(graph, *dq_node);

    // Replace target node's input (i.e. DQ node's output) with Q node's input
    std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacement_defs;
    replacement_defs[dq_node_output_arg] = q_node_first_input_arg;
    const Node& const_target_node = node_unit->GetNode();
    auto target_node = graph.GetNode(const_target_node.Index());
    target_node->ReplaceDefs(replacement_defs);

    // Remove DQ output defs
    // Note: If we don't remove the unused node args here, once calling graph.Resolve will fail 
    // due to graph.BuildConnections founds node arg that is not a graph input, initializer, or output of a previous node.
    // We can't rely on graph.CleanUnusedInitializersAndNodeArgs because it's called after graph.BuildConnections.
    for (auto node_arg : dq_node->MutableOutputDefs()) {
      graph.RemoveNodeArg(node_arg->Name());
    }
  }

  std::unordered_set<Node*> seen_q_nodes;

  // Remove Q output edges and node args
  for (auto qdq_pair : candidate_dq_to_q_map) {
    auto dq_node = qdq_pair.first;
    auto q_node = qdq_pair.second;
    graph.RemoveNode(dq_node->Index());

    if (seen_q_nodes.find(q_node) != seen_q_nodes.end()) {
      continue;
    }
    seen_q_nodes.insert(q_node);
    
    graph_utils::RemoveNodeOutputEdges(graph, *q_node);

    // Remove Q output defs
    // Note: If we don't remove the unused node args here, once calling graph.Resolve will fail
    // due to graph.BuildConnections founds node arg that is not a graph input, initializer, or output of a previous node.
    // We can't rely on graph.CleanUnusedInitializersAndNodeArgs because it's called after graph.BuildConnections.
    for (auto node_arg : q_node->MutableOutputDefs()) {
      graph.RemoveNodeArg(node_arg->Name());
    }
    graph.RemoveNode(q_node->Index());
    modified = true;
  }
  return Status::OK();
}

/**
 * Check whether the DQ node in the node unit and its producer Q node is the candidate QDQ pair, e.g. node_1 -> Q -> DQ -> node_2.
 * (Note: The Q and DQ are not in the same node unit)
 *
 *
 * There are no other rules except "it's a QDQ node pair" as well as node_index set for this optimizer to remove QDQ node pairs.
 * EP/caller should provide a specific set of Q/DQ nodes that are qualified to be removed.
 */
Status QDQStripping::FindCandidateQDQToRemove(Graph& graph, const NodeUnit& node_unit, std::unordered_map<Node*, Node*>& candidate_dq_to_q_map) const {
  assert(node_unit.UnitType() == NodeUnit::Type::QDQGroup);

  // Starts from DQ nodes in node unit
  for (auto const_dq_node : node_unit.GetDQNodes()) {
    if (!AllowStripping(*const_dq_node)) {
      continue;
    }
    auto node = const_dq_node->InputNodesBegin();
    bool qdq_removable = false;

    // If it's Q -> DQ and Q doesn't consume graph input. Then both Q and DQ can be removed.
    if (node != const_dq_node->InputNodesEnd() && node->OpType() == "QuantizeLinear" && node->GetInputEdgesCount() != 0) {
      qdq_removable = true;
    }

    if (!qdq_removable) {
      continue;
    }

    // Get mutable Q and DQ node
    auto q_node = graph.GetNode(node->Index());
    auto dq_node = graph.GetNode(const_dq_node->Index());
    candidate_dq_to_q_map[dq_node] = q_node;
  }
  return Status::OK();
}

/**
 * Returns true if the Q/DQ node can be removed, otherwise returns false.
 * If node_index set is not specified, any Q/DQ nodes can be removed.
 */
bool QDQStripping::AllowStripping(const Node& node) const {
  if (node.OpType() != "QuantizeLinear" && node.OpType() != "DequantizeLinear") {
    return false;
  }

  if (node_index_set_.empty()) {
    return true;
  }

  if (node_index_set_.find(node.Index()) != node_index_set_.end()) {
    return true;
  }
  return false;
}

Status QDQStripping::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                             const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);

  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer, logger);
  std::unordered_set<const NodeUnit*> seen_node_units;
  
  // The Q/DQ pairs selected by this optimizer to remove, e.g. node_1 -> Q -> DQ -> node_2
  // The reason to use a map here is that every DQ is unique due to
  // EnsureUniqueDQForNodeUnit optimization, but differetn DQ can trace back to the same Q
  std::unordered_map<Node*, Node*> candidate_dq_to_q_map;

  // Process node units in topological order
  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    gsl::not_null<const onnxruntime::Node*> node(graph_viewer.GetNode(node_index));
    
    // Get the node_unit associated with the node.
    gsl::not_null<const NodeUnit*> node_unit = node_unit_map.at(node);

    // Visiting 'nodes' in topological order does not guarantee that 'node_units' are
    // also visited in topological order. Skip this node if it is not the node_unit's target node
    // to ensure actual 'node_units' are visited in topological order.
    if (node != &node_unit->GetNode()) {
      continue;
    }

    if (seen_node_units.count(node_unit) != 0) {
      continue;  // Already handled this node unit
    }

    if (node_unit->UnitType() == NodeUnit::Type::QDQGroup) {
      FindCandidateQDQToRemove(graph, *node_unit, candidate_dq_to_q_map);
    }
  }

  RemoveQDQ(graph, candidate_dq_to_q_map, node_unit_map, modified);

  if (modified) {
    ORT_RETURN_IF_ERROR(graph.Resolve());
  }

  return Status::OK();
}

}  // namespace onnxruntime
