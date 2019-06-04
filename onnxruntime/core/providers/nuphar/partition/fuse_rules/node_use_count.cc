// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_use_count.h"
#include "core/codegen/common/common.h"
#include "core/providers/nuphar/common/analysis/graph_partition_stats.h"
#include "core/providers/nuphar/partition/fuse/fuse.h"
#include "core/providers/nuphar/common/nuphar_settings.h"

namespace onnxruntime {
namespace nuphar {

// Heuristics to determine if a node is valid for cut, i.e., we can create a subgraph
// for this node and its predecessors if they haven't been included into other subgraphs.
bool RuleNodeUseCount::CanCut(const onnxruntime::Node* node) const {
  return codegen::Promote<codegen::GraphPartitionStats>(graph_stats_)->NodeUseCount(node) > uses_valid_for_cut_;
}

void RuleNodeUseCount::AddToSubgraph(const onnxruntime::GraphViewer& graph,
                                     std::unique_ptr<IndexedSubGraph>& subgraph,
                                     std::set<NodeIndex>& claimed_nodes,
                                     int* acc_uses,
                                     std::vector<std::unique_ptr<ComputeCapability>>& result) {
  auto node_cnt = subgraph->nodes.size();
  if (node_cnt > 1) {
    if (codegen::CodeGenSettings::Instance().HasOption(nuphar_codegen::kNupharDumpFusedNodes)) {
      std::ostringstream stream;
      stream << "[NUPHAR_DUMP_FUSED_NODES]" << std::endl;
      stream << "Subgraph of size " << node_cnt << " [";
      for (const auto& node_index : subgraph->nodes) {
        const Node* node = graph.GetNode(node_index);
        stream << "(" << node->Name() << ", " << node->OpType()
               << ", " << codegen::Promote<codegen::GraphPartitionStats>(graph_stats_)->NodeUseCount(node) << ") ";
      }
      LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << stream.str();
    }

    // Safety check to avoid case of all nodes in subgraph are Identity nodes,
    // e.g. mlnet.onnx, for two reasons: 1) better performance, Identity will
    // be no-op in CPU due to name alias but will become reshape op in Nuphar.
    // and 2) incorrect schedule, (true) input of Identity will be scheduled
    // inline, which shouldn't since only compute op could be inlined.
    bool allIdentityNode = true;
    for (const auto& node_index : subgraph->nodes) {
      const Node* node = graph.GetNode(node_index);
      if (node->OpType() != "Identity") {
        allIdentityNode = false;
      }
    }

    if (!allIdentityNode) {
      claimed_nodes.insert(subgraph->nodes.begin(), subgraph->nodes.end());

      result.emplace_back(
          ToCapacity(
              graph,
              subgraph));
    }
  }

  subgraph.reset(new IndexedSubGraph());
  *acc_uses = 0;
}

Status RuleNodeUseCount::Fuse(const onnxruntime::GraphViewer& graph,
                              IsOpTypeSupportedFunc is_op_type_supported_func,
                              std::set<NodeIndex>& claimed_nodes,
                              std::vector<std::unique_ptr<ComputeCapability>>& result) {
  // Partition the graph (fusing ops) based on the heuristics used in Model Compiler:
  // (1) each node is assgined a value based on approximately how many times it's going
  //     to be used. For each input node of MatMul and Gemm ops, its use count is computed
  //     as the times it's used for matrix multiplication;
  // (2) In the course of graph traversal, we accumulate the uses for all the nodes that have
  //     been visited. When this accumulated use count is larger than or equal to a pre-defined
  //     threshold value (uses_threshold_), we create a subgraph if the current node is valid for cut;
  // (3) the criteria to determine if a node is valid for cut:
  //       * either the node's use count is larger than or equal to a pre-defined magic
  //         number (uses_valid_for_cut_); or
  //       * the node is an input to a "loop" node such as LSTM, RNN ops;
  // (4) those two pre-defined magic numbers are the same as what we used in Model Compiler.
  //     They are used for estimating the loop working-set size and by no mean to be optimal.
  //     More importantly, we should use a more sophistic approach to estimating the loop
  //     working-set size, which will be used for making fusion decisions.
  // (5) any LSTM/RNN/GRU op will be placed into its own subgraph, i.e. a subgraph that only
  //     contains this LSTM/RNN/GRU op.

  std::unique_ptr<IndexedSubGraph> subgraph = std::make_unique<IndexedSubGraph>();
  int acc_uses = 0;
  for (auto& node : graph.Nodes()) {
    // for Scan node, make sure its subgraph can be handled in Nuphar
    if (node.OpType() == "Scan") {
      for (const auto& scan_node : node.GetGraphAttribute("body")->Nodes()) {
        if (scan_node.GetFunctionBody()) {
          // unbox Function node
          for (const auto& func_node : scan_node.GetFunctionBody()->Body().Nodes()) {
            if (!is_op_type_supported_func(func_node))
              ORT_NOT_IMPLEMENTED("Unsupported OpType:", func_node.OpType(), " in Scan");
            else if (IsRecurrentNode(func_node))
              ORT_NOT_IMPLEMENTED("Recurrent node inside Scan is not supportet yet");
          }
        } else if (!is_op_type_supported_func(scan_node))
          ORT_NOT_IMPLEMENTED("Unsupported OpType:", scan_node.OpType(), " in Scan");
        else if (IsRecurrentNode(scan_node))
          ORT_NOT_IMPLEMENTED("Recurrent node inside Scan is not supportet yet");
      }
    }

    bool is_loop_node = IsRecurrentNode(node);
    bool is_supported_node = is_op_type_supported_func(node);
    if (is_loop_node || !is_supported_node) {
      // fuse previously-collected nodes
      AddToSubgraph(graph, subgraph, claimed_nodes, &acc_uses, result);
      if (is_loop_node && is_supported_node) {
        // put nodes such as LSTM into their own subgraphs
        subgraph->nodes.push_back(node.Index());
        AddToSubgraph(graph, subgraph, claimed_nodes, &acc_uses, result);
      }
      continue;
    }

    subgraph->nodes.push_back(node.Index());
    int use_cnt = codegen::Promote<codegen::GraphPartitionStats>(graph_stats_)->NodeUseCount(&node);
    ORT_ENFORCE(use_cnt > 0);
    acc_uses += use_cnt;
    if (acc_uses < uses_threshold_) {
      continue;
    }

    if (CanCut(&node)) {
      AddToSubgraph(graph, subgraph, claimed_nodes, &acc_uses, result);
    }
  }

  // In case we have any node left
  AddToSubgraph(graph, subgraph, claimed_nodes, &acc_uses, result);
  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
