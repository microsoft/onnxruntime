// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/op.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/promote_topo_order_rewriter.h"

namespace onnxruntime {

static bool IsBackwardNode(const Node& node) {
  return node.Description() == "Backward pass";
}

static std::unordered_set<std::string> nodes_to_promote = {
    "SwapToHost",
    "Shape"};

std::vector<std::string> PromoteTopologicalOrderRewriter::TargetOpTypes() const noexcept {
  return std::vector<std::string>(nodes_to_promote.begin(), nodes_to_promote.end());  // make sure these nodes happen as early as possible
}

bool PromoteTopologicalOrderRewriter::SatisfyCondition(const Graph& /*graph*/, const Node& node, const logging::Logger& /*logger*/) const {
  if (node.GetInputEdgesCount() == 0)
    return false;

  // check if the node already have control edge
  for (auto iter = node.OutputEdgesBegin(); iter != node.OutputEdgesEnd(); ++iter) {
    if (iter->IsControlEdge())
      return false;
  }
  return true;
}

Status PromoteTopologicalOrderRewriter::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  std::unordered_map<NodeIndex, int> topo_indices;
  GraphViewer graph_viewer(graph);
  int topo_index = 0;
  topo_indices.clear();
  for (const auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    topo_indices.insert(std::make_pair(index, topo_index++));
  }
  AddEdgeInForward(graph, node, topo_indices);
  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  return Status::OK();
}

void PromoteTopologicalOrderRewriter::AddEdgeInForward(Graph& graph, Node& node, const std::unordered_map<NodeIndex, int>& topo_indices) const {
  // need to make sure node happens as early as possible in topological sort
  // find the src_node to node, and then find its output as dst_node
  // and then add an edge from node to dst_node to make sure node happens before dst_node
  const auto& src_edge = *(node.InputEdgesBegin());
  const auto& src_node = src_edge.GetNode();
  const auto& src_arg_idx = src_edge.GetSrcArgIndex();

  NodeIndex node_idx = node.Index();
  int min_topo_index = INT_MAX;
  NodeIndex node_found = 0;
  int min_arg_topo_index = INT_MAX;
  NodeIndex arg_node_found = 0;

  // add control edge
  for (auto iter = src_node.OutputEdgesBegin(); iter != src_node.OutputEdgesEnd(); ++iter) {
    const Node& peer_node = iter->GetNode();
    if (nodes_to_promote.count(peer_node.OpType()))
      continue;

    int topo_index = topo_indices.at(peer_node.Index());
    if (iter->GetSrcArgIndex() == src_arg_idx) {
      if (topo_index < min_topo_index) {
        min_topo_index = topo_index;
        node_found = peer_node.Index();
      }
    } else if (!IsBackwardNode(iter->GetNode())) {
      if (topo_index < min_arg_topo_index) {
        min_arg_topo_index = topo_index;
        arg_node_found = peer_node.Index();
      }
    }
  }
  // add new edge to enforce swap node order, and update precedences
  if (min_topo_index < INT_MAX) {
    graph.AddControlEdge(node_idx, node_found);
  } else if (min_arg_topo_index < INT_MAX) {
    graph.AddControlEdge(node_idx, arg_node_found);
  } else if (node.OpType() == "SwapToHost") {
    // there could be some optimizations making src_node no longer needed in FW, while used in BW
    // so remove swap nodes from src_node, and link src_node directly to dst_node
    const auto& swap_in = *node.OutputNodesBegin();
    Node::EdgeSet swap_in_output_edges(swap_in.OutputEdgesBegin(), swap_in.OutputEdgesEnd());
    for (auto edge : swap_in_output_edges) {
      graph.RemoveEdge(swap_in.Index(), edge.GetNode().Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
      graph.AddEdge(src_node.Index(), edge.GetNode().Index(), src_arg_idx, edge.GetDstArgIndex());
    }
    graph.RemoveNode(swap_in.Index());
    graph.RemoveNode(node_idx);
  }
}

}  // namespace onnxruntime
