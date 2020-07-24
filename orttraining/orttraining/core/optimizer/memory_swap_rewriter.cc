// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/op.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/memory_swap_rewriter.h"

namespace onnxruntime {

static bool IsBackwardNode(const Node& node) {
  return node.Description() == "Backward pass";
}

static bool ShouldHandleSrcNode(const Node& node) {
  // blacklist some ops for memory swap
  // TODO: make it configurable
  static const std::unordered_set<std::string> ignore_src_op_types =
      {"Shape"};
  return !IsBackwardNode(node) && 0 == ignore_src_op_types.count(node.OpType());
}

static bool ShouldHandleDstNode(const Node& node) {
  // blacklist ops and arg_idx for memory swap
  static const std::unordered_set<std::string> ignore_dst_op_args =
      {"Shape"};
  return IsBackwardNode(node) && 0 == ignore_dst_op_args.count(node.OpType());
}

Status MemorySwapRewriter::Apply(Graph& graph, Node& src_node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  std::unordered_set<int> to_bw_arg_idx;
  for (auto edge_iter = src_node.OutputEdgesBegin(); edge_iter != src_node.OutputEdgesEnd(); ++edge_iter) {
    if (ShouldHandleDstNode(edge_iter->GetNode())) {
      to_bw_arg_idx.insert(edge_iter->GetSrcArgIndex());
    }
  }
  for (int src_node_output_idx : to_bw_arg_idx) {
    NodeArg* src_node_output_arg = const_cast<NodeArg*>(src_node.OutputDefs()[src_node_output_idx]);
    auto& swap_out_arg = graph.GetOrCreateNodeArg(src_node_output_arg->Name() + "_memswap_out", src_node_output_arg->TypeAsProto());
    auto& swap_in_arg = graph.GetOrCreateNodeArg(src_node_output_arg->Name() + "_memswap_in", src_node_output_arg->TypeAsProto());
    auto& swap_out_node = graph.AddNode(src_node_output_arg->Name() + "_swapout",
                                        "SwapToHost",
                                        "",
                                        {src_node_output_arg},
                                        {&swap_out_arg});
    auto& swap_in_node = graph.AddNode(src_node_output_arg->Name() + "_swapin",
                                       "SwapFromHost",
                                       "Backward pass",
                                       {&swap_out_arg},
                                       {&swap_in_arg});

    // process output edges from this output_def
    // note this needs to happen before linking src_node with swap_out_node
    // and since the operation might change src_node's OutputEdges, needs a copy of original edges
    Node::EdgeSet src_node_output_edges(src_node.OutputEdgesBegin(), src_node.OutputEdgesEnd());
    for (const auto& output_edge : src_node_output_edges) {
      if (output_edge.GetSrcArgIndex() != src_node_output_idx)
        continue;

      if (!ShouldHandleDstNode(output_edge.GetNode()))
        continue;

      const Node& dst_node = output_edge.GetNode();
      int dst_arg_idx = output_edge.GetDstArgIndex();
      // remove edge from src_node to dst_node
      graph.RemoveEdge(src_node.Index(), dst_node.Index(), src_node_output_idx, dst_arg_idx);
      // add edge from swap_in to dst_node
      graph.AddEdge(swap_in_node.Index(), dst_node.Index(), 0, dst_arg_idx);
    }

    // add edges in graph
    graph.AddEdge(src_node.Index(), swap_out_node.Index(), src_node_output_idx, 0);
    graph.AddEdge(swap_out_node.Index(), swap_in_node.Index(), 0, 0);
  }

  // after adding nodes, rerun topological sort
  need_topo_sort_ = true;

  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  return Status::OK();
}

bool MemorySwapRewriter::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& /*logger*/) const {
  // only check forward nodes
  if (!ShouldHandleSrcNode(node))
    return false;

  // update topological sort results
  need_topo_sort_ = need_topo_sort_ || (last_graph_ != &graph);
  if (need_topo_sort_) {
    last_graph_ = &graph;
    GraphViewer graph_viewer(graph);
    int topo_index = 0;
    topo_indices_.clear();
    for (const auto index : graph_viewer.GetNodesInTopologicalOrder()) {
      topo_indices_.insert(std::make_pair(index, topo_index));
      const Node* p = graph_viewer.GetNode(index);
      p->ForEachWithIndex(
          p->OutputDefs(),
          [this, topo_index](const NodeArg& arg, size_t) {
            if (arg.Name() == stop_at_node_arg_)
              stop_at_topo_index_ = topo_index;
            return Status::OK();
          });
      ++topo_index;
    }
  }

  int topo_idx = topo_indices_.at(node.Index());
  if (topo_idx > stop_at_topo_index_)
    return false;

  for (auto iter = node.OutputEdgesBegin(); iter != node.OutputEdgesEnd(); ++iter) {
    if (ShouldHandleDstNode(iter->GetNode())) {
      return true;
    }
  }
  return false;
}

}  // namespace onnxruntime
