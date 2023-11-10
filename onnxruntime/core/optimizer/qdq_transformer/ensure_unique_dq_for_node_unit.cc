// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/ensure_unique_dq_for_node_unit.h"

#include <cassert>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {

namespace {

constexpr auto* kTransformerName = "EnsureUniqueDQForNodeUnit";

// Given `original_dq_output_edge`, an edge from DQ to an input of Y, duplicate DQ to a new node, DQ'.
// After duplication, the input of Y from `original_dq_output_edge` is the only consumer of DQ'.
//
// Convert this: <DQ_inputs> -> DQ -> ...
//                              |
//                              +---> Y
//
// To this:      <DQ_inputs> -> DQ -> ...
//                    |
//                    +-------> DQ' -> Y
//
// Note: <DQ_inputs> may be graph inputs/initializers or nodes.
Status DuplicateDQForOutputEdge(const graph_utils::GraphEdge& original_dq_output_edge, Graph& graph) {
  // DQ
  Node* original_dq_node_ptr = graph.GetNode(original_dq_output_edge.src_node);
  assert(original_dq_node_ptr != nullptr);
  Node& original_dq_node = *original_dq_node_ptr;

  // Y
  Node* dst_node_ptr = graph.GetNode(original_dq_output_edge.dst_node);
  assert(dst_node_ptr != nullptr);
  Node& dst_node = *dst_node_ptr;

  // set up new NodeArg
  auto& new_dq_output_nodearg =
      graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(original_dq_output_edge.arg_name + "/duplicated"), nullptr);

  // set up new Node
  const auto& dq_inputs = original_dq_node.MutableInputDefs();

  // DQ'
  Node& new_dq_node = graph.AddNode(graph.GenerateNodeName(original_dq_node.Name() + "/duplicated"),
                                    QDQ::DQOpName,
                                    MakeString("Added by ", kTransformerName),
                                    dq_inputs,
                                    {&new_dq_output_nodearg});

  // set up edges
  // remove DQ -> Y
  graph_utils::GraphEdge::RemoveGraphEdges(graph, {original_dq_output_edge});

  // add DQ_input -> DQ' if DQ_input is a node
  const auto original_dq_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(original_dq_node);
  for (const auto& original_dq_input_edge : original_dq_input_edges) {
    // create edge from the original DQ node's input node
    graph.AddEdge(original_dq_input_edge.src_node, new_dq_node.Index(),
                  original_dq_input_edge.src_arg_index, original_dq_input_edge.dst_arg_index);
  }

  // add DQ' -> Y
  dst_node.MutableInputDefs()[original_dq_output_edge.dst_arg_index] = &new_dq_output_nodearg;
  graph.AddEdge(new_dq_node.Index(), original_dq_output_edge.dst_node, 0, original_dq_output_edge.dst_arg_index);

  return Status::OK();
}

Status EnsureUniqueDQForEachExplicitOutputEdge(const Node& node, Graph& graph, bool& modified) {
  if (!QDQ::MatchDQNode(node)) {
    return Status::OK();
  }

  const auto& dq_node = node;

  // QDQ node units are only formed by nodes in the current graph, and not between nodes in the current graph and a
  // subgraph. Consequently, we only duplicate DQ's for edges to explicit inputs and skip edges to implicit (subgraph)
  // inputs.

  const bool produces_graph_output = graph.NodeProducesGraphOutput(node);

  auto dq_output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(dq_node, 0);

  // Check for common cases where we don't need to do anything:
  // 1. DQ has no output edges.
  //
  // It may have a graph output. Either way, there is no same-graph consumer node input to consider.
  //
  // 2. DQ has exactly one output edge and no graph output.
  //
  // If the output edge is to an implicit input, there are only consumer nodes in a subgraph which we can ignore.
  // Otherwise, the output edge is to an explicit input of the only same-graph consumer node so the DQ is already
  // unique.
  if (dq_output_edges.empty() ||
      (dq_output_edges.size() == 1 && !produces_graph_output)) {
    return Status::OK();
  }

  // Remove edges to implicit inputs from consideration.
  const auto dq_output_edges_to_explicit_inputs_end = std::remove_if(
      dq_output_edges.begin(), dq_output_edges.end(),
      [&const_graph = std::as_const(graph)](const graph_utils::GraphEdge& dq_output_edge) {
        const Node* consumer_node_ptr = const_graph.GetNode(dq_output_edge.dst_node);
        assert(consumer_node_ptr != nullptr);
        const Node& consumer_node = *consumer_node_ptr;
        const auto consumer_explicit_input_defs_count = consumer_node.InputDefs().size();
        return narrow<size_t>(dq_output_edge.dst_arg_index) >= consumer_explicit_input_defs_count;
      });

  const bool has_subgraph_consumer = dq_output_edges_to_explicit_inputs_end != dq_output_edges.end();
  if (has_subgraph_consumer) {
    dq_output_edges.erase(dq_output_edges_to_explicit_inputs_end, dq_output_edges.end());
  }

  // If the original DQ produces a graph output or has a subgraph consumer node, we preserve any of those output
  // relationships and duplicate new unique DQ's for each of the same-graph consumer nodes. It is important that the
  // original DQ node does not become part of a QDQ node unit as it has an additional consumer.
  //
  // Otherwise, where the original DQ has only same-graph consumer nodes, we can reuse the original DQ as the unique
  // DQ for the first same-graph consumer node and duplicate new unique DQ's for the rest of them.
  const bool must_preserve_original_dq_outside_of_node_unit = produces_graph_output || has_subgraph_consumer;

  auto edge_to_process = dq_output_edges.begin();
  if (!must_preserve_original_dq_outside_of_node_unit && edge_to_process != dq_output_edges.end()) {
    // reuse original DQ for the first edge, start duplicating from the next edge
    ++edge_to_process;
  }

  for (; edge_to_process != dq_output_edges.end(); ++edge_to_process) {
    ORT_RETURN_IF_ERROR(DuplicateDQForOutputEdge(*edge_to_process, graph));
    modified = true;
  }

  return Status::OK();
}

}  // namespace

Status EnsureUniqueDQForNodeUnit::ApplyImpl(Graph& graph,
                                            bool& modified,
                                            int graph_level,
                                            const logging::Logger& logger) const {
  const GraphViewer graph_viewer{graph};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    auto* node_ptr = graph.GetNode(node_idx);
    if (!node_ptr) {
      continue;
    }

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    ORT_RETURN_IF_ERROR(EnsureUniqueDQForEachExplicitOutputEdge(node, graph, modified));
  }

  return Status::OK();
}

EnsureUniqueDQForNodeUnit::EnsureUniqueDQForNodeUnit()
    : GraphTransformer{kTransformerName, {}} {
}

}  // namespace onnxruntime
