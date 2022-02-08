// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_util.h"

#include <utility>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"

namespace onnxruntime::QDQ {

bool MatchQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, QOpName, {10, 13});
}

bool MatchDQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, DQOpName, {10, 13});
}

static Status RemoveDQQPair(Graph& graph, Node& dq_node, Node& q_node) {
  // remove edge between parent of DQ to DQ
  std::pair<NodeIndex, int> input_edge_info{0, -1};
  auto* dq_input_edge_0 = graph_utils::GetInputEdge(dq_node, 0);
  if (dq_input_edge_0) {
    input_edge_info.first = dq_input_edge_0->GetNode().Index();
    input_edge_info.second = dq_input_edge_0->GetSrcArgIndex();
    graph.RemoveEdge(dq_input_edge_0->GetNode().Index(), dq_node.Index(),
                     dq_input_edge_0->GetSrcArgIndex(), dq_input_edge_0->GetDstArgIndex());
  }

  graph_utils::RemoveNodeOutputEdges(graph, dq_node);  // Remove DQ node output edges

  auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(q_node, 0);
  graph_utils::RemoveNodeOutputEdges(graph, q_node);  // Remove Q node output edges
  for (auto& output_edge : output_edges) {
    // set input NodeArg of Q's children to the 1st input of DQ
    graph.GetNode(output_edge.dst_node)->MutableInputDefs()[output_edge.dst_arg_index] =
        dq_node.MutableInputDefs()[0];

    // add edge between parent of DQ to children of Q
    if (input_edge_info.second != -1) {
      graph.AddEdge(input_edge_info.first, output_edge.dst_node,
                    input_edge_info.second, output_edge.dst_arg_index);
    }
  }

  ORT_RETURN_IF_NOT(graph.RemoveNode(dq_node.Index()), "Failed to remove DQ node at index ", dq_node.Index());
  ORT_RETURN_IF_NOT(graph.RemoveNode(q_node.Index()), "Failed to remove Q node at index ", q_node.Index());

  return Status::OK();
}

Status CancelOutRedundantDQQPairs(Graph& graph, gsl::span<const NodeIndex> node_indices,
                                  const std::unordered_set<std::string>& compatible_providers,
                                  const logging::Logger& logger,
                                  bool& modified) {
  auto get_const_initializer = [&graph](const std::string& initializer_name) {
    return graph.GetConstantInitializer(initializer_name, true);
  };

  for (NodeIndex node_idx : node_indices) {
    Node* node = graph.GetNode(node_idx);
    if (!node) {
      continue;
    }

    Node& dq_node = *node;
    if (!QDQ::MatchDQNode(dq_node) ||
        !graph_utils::IsSupportedProvider(dq_node, compatible_providers) ||
        // check that DQ has one output edge, does not produce graph output
        !optimizer_utils::CheckOutputEdges(graph, dq_node, 1)) {
      continue;
    }

    Node& q_node = *graph.GetNode(dq_node.OutputNodesBegin()->Index());
    if (!QDQ::MatchQNode(q_node) ||
        !graph_utils::IsSupportedProvider(q_node, compatible_providers) ||
        // check that Q does not produce graph output
        graph.NodeProducesGraphOutput(q_node)) {
      continue;
    }

    // TODO requires scalar scale and zero point which may be stricter than needed
    if (!IsQDQPairSupported(q_node, dq_node, get_const_initializer, graph.ModelPath())) {
      continue;
    }

    LOGS(logger, VERBOSE) << "Removing redundant DQ/Q pair: "
                          << MakeString("(\"", dq_node.Name(), "\", index: ", dq_node.Index(), ")")
                          << " -> "
                          << MakeString("(\"", q_node.Name(), "\", index: ", q_node.Index(), ")");
    ORT_RETURN_IF_ERROR(RemoveDQQPair(graph, dq_node, q_node));
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime::QDQ
