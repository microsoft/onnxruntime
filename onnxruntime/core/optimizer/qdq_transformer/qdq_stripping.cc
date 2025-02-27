// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_stripping.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/framework/node_unit.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {
/**
 * Remove Q and DQ node pairs in the graph.
 *
 * @param candidate_dq_to_q_map All the qualified Q and DQ node pairs to be removed.
 */
Status RemoveQDQ(Graph& graph,
                 std::unordered_map<Node*, Node*>& candidate_dq_to_q_map,
                 std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                 bool& modified) {
  // Prior to remove the Q and DQ nodes, we need to 'rewire' input or output of the nodes.
  // Two scenarios:
  // 1. Replace target node's input (i.e. DQ node's output) with Q node's input.
  //    i.e. node_1 -> Q -> DQ -> node_2  => node_1 -> node_2
  //         graph's input -> Q -> DQ -> node_1 => graph's input -> node_1
  //
  // 2. Make producer node of Q node generate the graph's output.
  //    i.e. node_1 -> Q -> DQ -> graph's output => node_1 -> graph's output
  //
  std::map<onnxruntime::Node*, std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*>> node_to_replacement_defs_map;

  // Find src node arg to be replaced by dst node arg
  for (auto qdq_pair : candidate_dq_to_q_map) {
    auto dq_node = qdq_pair.first;
    auto q_node = qdq_pair.second;
    assert(node_unit_map.find(dq_node) != node_unit_map.end());
    auto node_unit = node_unit_map[dq_node];

    assert(q_node->InputDefs().size() == 3);
    assert(dq_node->OutputDefs().size() == 1);
    auto q_node_input_arg = q_node->MutableInputDefs()[0];
    auto dq_node_output_arg = dq_node->MutableOutputDefs()[0];

    std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacement_defs;
    auto target_node = graph.GetNode(node_unit->GetNode().Index());
    if (target_node != dq_node) {
      // Scenario #1: node_1 -> Q -> DQ -> node_2  => node_1 -> node_2
      if (node_to_replacement_defs_map.find(target_node) == node_to_replacement_defs_map.end()) {
        replacement_defs[dq_node_output_arg] = q_node_input_arg;
        node_to_replacement_defs_map[target_node] = replacement_defs;
      } else {
        node_to_replacement_defs_map[target_node][dq_node_output_arg] = q_node_input_arg;
      }
    } else {
      // Scenario #2: node_1 -> Q -> DQ -> graph's output => node_1 -> graph's output
      auto node = q_node->InputNodesBegin();
      if (node != q_node->InputNodesEnd()) {
        target_node = graph.GetNode(node->Index());
        if (node_to_replacement_defs_map.find(target_node) == node_to_replacement_defs_map.end()) {
          replacement_defs[q_node_input_arg] = dq_node_output_arg;
          node_to_replacement_defs_map[target_node] = replacement_defs;
        } else {
          node_to_replacement_defs_map[target_node][q_node_input_arg] = dq_node_output_arg;
        }
      }
    }
  }

  std::unordered_set<Node*> seen_q_nodes;
  std::unordered_set<std::string> node_args_to_remove;  // The node args of Q/DQ nodes that no longer needed

  // Remove Q/DQ nodes
  for (auto qdq_pair : candidate_dq_to_q_map) {
    auto dq_node = qdq_pair.first;
    auto q_node = qdq_pair.second;

    // Collect DQ nodes' node args to be removed at the end
    // Make sure not to include graph's output
    for (auto node_arg : dq_node->MutableOutputDefs()) {
      bool is_graph_output = false;
      for (auto graph_output : graph.GetOutputs()) {
        if (node_arg == graph_output) {
          is_graph_output = true;
          break;
        }
      }
      if (!is_graph_output) {
        node_args_to_remove.insert(node_arg->Name());
      }
    }

    // Remove DQ output edges and DQ node itself
    graph_utils::RemoveNodeOutputEdges(graph, *dq_node);
    graph.RemoveNode(dq_node->Index());

    if (seen_q_nodes.find(q_node) != seen_q_nodes.end()) {
      continue;
    }
    seen_q_nodes.insert(q_node);

    // Collect Q nodes' node args to be removed at the end
    for (auto node_arg : q_node->MutableOutputDefs()) {
      node_args_to_remove.insert(node_arg->Name());
    }

    // Remove Q output edges and Q node itself
    graph_utils::RemoveNodeOutputEdges(graph, *q_node);
    graph.RemoveNode(q_node->Index());

    modified = true;
  }

  // Replace src node arg with dst node arg
  for (auto entry : node_to_replacement_defs_map) {
    auto target_node = entry.first;
    auto replacement_defs = entry.second;
    target_node->ReplaceDefs(replacement_defs);
  }

  // Remove all the unused node args
  //
  // Note: If we don't remove the unused node args here, once calling graph.Resolve will fail
  // due to graph.BuildConnections founds node arg that is not a graph input, initializer, or output of a previous node.
  // We can't rely on graph.CleanUnusedInitializersAndNodeArgs because it's called after graph.BuildConnections.
  for (auto node_arg : node_args_to_remove) {
    graph.RemoveNodeArg(node_arg);
  }

  return Status::OK();
}

/**
 * Check whether the DQ node in the node unit and its producer Q node, e.g. node_1 -> Q -> DQ -> node_2, are qualified to be removed.
 * (Note: The Q and DQ are not in the same node unit)
 *
 * There are no other rules except "it's a QDQ node pair" as well as node_index set in AllowStripping.
 * EP should provide a set of Q/DQ nodes, i.e. node_index set, that are qualified to be removed.
 */
Status QDQStripping::FindCandidateQDQToRemove(Graph& graph, const NodeUnit& node_unit, std::unordered_map<Node*, Node*>& candidate_dq_to_q_map) const {
  std::vector<const Node*> dq_nodes;

  // Starts from DQ nodes in node unit
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    for (auto const_dq_node : node_unit.GetDQNodes()) {
      dq_nodes.push_back(const_dq_node);
    }
  } else {
    const auto& node = node_unit.GetNode();
    if (node.OpType() == "DequantizeLinear") {
      dq_nodes.push_back(&node);
    }
  }

  for (auto const_dq_node : dq_nodes) {
    if (!AllowStripping(*const_dq_node)) {
      continue;
    }

    auto node = const_dq_node->InputNodesBegin();
    bool qdq_removable = false;

    // If it's Q -> DQ, then both Q and DQ can be removed.
    if (node != const_dq_node->InputNodesEnd() && node->OpType() == "QuantizeLinear") {
      qdq_removable = true;
    }

    if (!qdq_removable) {
      continue;
    }

    //  Get mutable Q and DQ node
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

  // DQ -> Q map where Q node is the producer node for DQ node, e.g. node_1 -> Q -> DQ -> node_2.
  // The DQ and Q pairs are selected by this optimizer to be removed.
  // The reason to use a map here is that every DQ is unique because of applying EnsureUniqueDQForNodeUnit optimization,
  // but different DQ nodes can trace back to the same producer Q node.
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
    seen_node_units.insert(node_unit);

    FindCandidateQDQToRemove(graph, *node_unit, candidate_dq_to_q_map);
  }

  RemoveQDQ(graph, candidate_dq_to_q_map, node_unit_map, modified);

  if (modified) {
    ORT_RETURN_IF_ERROR(graph.Resolve());
  }

  return Status::OK();
}

}  // namespace onnxruntime
