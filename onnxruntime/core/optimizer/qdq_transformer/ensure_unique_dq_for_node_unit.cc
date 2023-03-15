// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/ensure_unique_dq_for_node_unit.h"

#include "core/common/common.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {

namespace {

// Given `original_dq_output_edge`, the edge from DQ to Y, duplicate DQ so that Y has a new input node DQ', for which it
// is the only consumer.
//
// Convert this: X -> DQ -> ...
//                    |
//                    +---> Y
//
// To this:      X -> DQ -> ...
//               |
//               +--> DQ' -> Y
//
// Note: Y should be a node, not an output. X may be either an input/initializer or a node.
Status DuplicateDQForEdge(const graph_utils::GraphEdge& original_dq_output_edge, Graph& graph) {
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
                                    MakeString("Added by ", EnsureUniqueDQForNodeUnit::kTransformerName),
                                    dq_inputs,
                                    {&new_dq_output_nodearg});

  // set up edges
  // remove DQ -> Y
  graph_utils::GraphEdge::RemoveGraphEdges(graph, {original_dq_output_edge});

  // add X -> DQ' if X is a node
  if (const auto original_dq_input_edge = graph_utils::GraphEdge::GetNodeInputEdge(original_dq_node, 0);
      original_dq_input_edge.has_value()) {
    // create another edge from the original DQ node's input node
    graph.AddEdge(original_dq_input_edge->src_node, new_dq_node.Index(), original_dq_input_edge->src_arg_index, 0);
  }

  // add DQ' -> Y
  dst_node.MutableInputDefs()[original_dq_output_edge.dst_arg_index] = &new_dq_output_nodearg;
  graph.AddEdge(new_dq_node.Index(), original_dq_output_edge.dst_node, 0, original_dq_output_edge.dst_arg_index);

  return Status::OK();
}

Status EnsureUniqueDQForEachConsumerNode(const Node& node, Graph& graph, bool& modified) {
  if (!QDQ::MatchDQNode(node)) {
    return Status::OK();
  }

  const auto& dq_node = node;

  auto dq_output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(dq_node, 0);

  // if node has graph output (graph.NodeProducesGraphOutput(node))
  //   or node has consumer in subgraph (check for consumer with implicit inputs),
  //   then do not reuse the existing DQ for the first consumer edge.
  //   otherwise, it can be reused and we only need to duplicate from the second consumer on.
  // should only duplicate for consumers in current graph (where DQ output is not an implicit input).

  // partition output edges: [edges to explicit inputs, edges to implicit inputs]
  const auto dq_output_edges_to_explicit_inputs_end = std::remove_if(
      dq_output_edges.begin(), dq_output_edges.end(),
      [&const_graph = std::as_const(graph)](const graph_utils::GraphEdge& dq_output_edge) {
        const Node* consumer_node_ptr = const_graph.GetNode(dq_output_edge.dst_node);
        assert(consumer_node_ptr != nullptr);
        const Node& consumer_node = *consumer_node_ptr;
        const auto input_defs_count = consumer_node.InputDefs().size();
        return dq_output_edge.dst_arg_index < input_defs_count;
      });

  const bool has_output_edge_to_implicit_input = dq_output_edges_to_explicit_inputs_end != dq_output_edges.end();

  if (has_output_edge_to_implicit_input) {
    dq_output_edges.erase(dq_output_edges_to_explicit_inputs_end, dq_output_edges.end());
  }

  const bool produces_graph_output = graph.NodeProducesGraphOutput(node);
  const bool is_original_dq_in_use = has_output_edge_to_implicit_input || produces_graph_output;

  auto next_edge_to_process = dq_output_edges.begin();
  if (!is_original_dq_in_use && next_edge_to_process != dq_output_edges.end()) {
    ++next_edge_to_process;
  }

  for (; next_edge_to_process != dq_output_edges.end(); ++next_edge_to_process) {
    ORT_RETURN_IF_ERROR(DuplicateDQForEdge(*next_edge_to_process, graph));
    modified = true;
  }

  return Status::OK();
}

}  // namespace

Status EnsureUniqueDQForNodeUnit::ApplyImpl(Graph& graph,
                                            bool& modified,
                                            int graph_level,
                                            const logging::Logger& logger) const {
  // basic idea
  // for each DQ node
  //   if there are multiple consumer nodes, assume that each consumer node is part of a separate node group
  //     ensure that each consumer node has its own DQ node, duplicating from this one as necessary

  GraphViewer graph_viewer{graph};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    auto* node_ptr = graph.GetNode(node_idx);
    if (!node_ptr) {
      continue;
    }

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    ORT_RETURN_IF_ERROR(EnsureUniqueDQForEachConsumerNode(node, graph, modified));
  }

  return Status::OK();
}

}  // namespace onnxruntime
