// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_final_cleanup.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/selectors_actions/actions.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

// Convert QuantizeLinear and DequantizeLinear pair with type int8_t to type uint8_t
Status QDQFinalCleanupTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                             const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // node removed as part of an earlier fusion

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // if current node is Q or DQ
    // see if one output edge and not graph output
    // if next node is opposite
    //    and is not a graph output
    // remove both
    bool is_q = false;
    bool is_dq = false;
    Node* next_node_ptr = nullptr;

    if ((is_q = QDQ::MatchQNode(node)) == true || (is_dq = QDQ::MatchDQNode(node)) == true) {
      if (graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) &&
          optimizer_utils::CheckOutputEdges(graph, node, 1)) {
        next_node_ptr = graph.GetNode(node.OutputNodesBegin()->Index());
      }
    }

    if (!next_node_ptr) {
      continue;
    }

    Node& next_node = *next_node_ptr;  // we know it exists as it can from an edge

    if ((is_q && QDQ::MatchDQNode(next_node)) ||
        (is_dq && QDQ::MatchQNode(next_node))) {
      // we have Q -> DQ or DQ -> D pair
      if (!graph_utils::IsSupportedProvider(next_node, GetCompatibleExecutionProviders()) ||
          !optimizer_utils::CheckOutputEdges(graph, next_node, 1)) {
        continue;
      }

      // src node -> DQ/Q pair in either order -> downstream node
      NodeArg* src_nodearg = node.MutableInputDefs()[0];

      NodeIndex src_node_idx = 0;
      int src_arg_idx = -1;
      NodeIndex downstream_node_idx = 0;
      int downstream_arg_idx = -1;

      // input could be node or initializer/graph input so need to handle both.
      // if it's an initializer or graph input we just use src_nodearg.
      // if it's a node we need to use info from the edge.
      const Node::EdgeEnd* input_edge = nullptr;
      if (node.GetInputEdgesCount() == 1) {
        input_edge = &*node.InputEdgesBegin();
        src_node_idx = input_edge->GetNode().Index();
        src_arg_idx = input_edge->GetSrcArgIndex();
        // remove edge from src to first node in pair. dest arg idx is 0 as Q/DQ only has one input
        graph.RemoveEdge(src_node_idx, node.Index(), src_arg_idx, 0);
      }

      // remove edge between pair we're removing
      // both DQ and Q are single input single output so src idx and dest idx must be 0
      graph.RemoveEdge(node.Index(), next_node.Index(), 0, 0);

      // remove edge to downstream node
      const Node::EdgeEnd& output_edge = *next_node.OutputEdgesBegin();
      downstream_node_idx = output_edge.GetNode().Index();
      downstream_arg_idx = output_edge.GetDstArgIndex();

      // source arg idx is 0 as Q/DQ only has one output
      graph.RemoveEdge(next_node.Index(), downstream_node_idx, 0, downstream_arg_idx);

      // replace input on downstream node
      Node& downstream_node = *graph.GetNode(downstream_node_idx);
      downstream_node.MutableInputDefs()[downstream_arg_idx] = src_nodearg;

      // create edge between src_node (if available) and downstream node
      if (input_edge) {
        graph.AddEdge(src_node_idx, downstream_node_idx, src_arg_idx, downstream_arg_idx);
      }

      graph.RemoveNode(node.Index());
      graph.RemoveNode(next_node.Index());

      modified = true;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
