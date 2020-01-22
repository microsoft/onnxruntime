// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/matmul_transpose_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

static std::pair<bool, Node*> IsInputTranspose(const Graph& graph, NodeArg& node_arg) {
  auto trans_node = graph.GetProducerNode(node_arg.Name());
  if (trans_node == nullptr || trans_node->OpType() != "Transpose") {
    return std::make_pair(false, nullptr);
  }

  auto perms = RetrieveValues<int64_t>(trans_node->GetAttributes().at("perm"));
  int64_t rank = perms.size();
  if (rank < 2) {
    return std::make_pair(false, nullptr);
  }

  bool is_trans_on_last_two_dims = true;
  for (int64_t i = 0; i < rank - 2; i++) {
    if ((int64_t)perms[i] != i) {
      is_trans_on_last_two_dims = false;
      break;
    }
  }

  if (is_trans_on_last_two_dims) {
    is_trans_on_last_two_dims = (int64_t)perms[rank - 2] == rank - 1 && (int64_t)perms[rank - 1] == rank - 2;
  }

  if (!is_trans_on_last_two_dims) {
    return std::make_pair(false, nullptr);
  }

  return std::make_pair(true, const_cast<Node*>(trans_node));
}

static size_t UpdateConsumerCount(Graph& graph, NodeArg* target, std::unordered_map<NodeArg*, size_t>& count_map) {
  const auto& node_consumers = graph.GetConsumerNodes(target->Name());
  ORT_ENFORCE(!node_consumers.empty());
  auto it = count_map.find(target);
  if (it == count_map.end()) {
    count_map.insert({target, node_consumers.size() - 1});
    return node_consumers.size() - 1;
  } else {
    count_map[target] -= 1;
    return count_map[target];
  }
}

Status MatmulTransposeFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::deque<onnxruntime::NodeIndex> removed_nodes;

  std::unordered_map<NodeArg*, size_t> consumer_count;

  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if ((!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {9}) &&
         !graph_utils::IsSupportedOptypeVersionAndDomain(node, "TransposeMatMul", {9})) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    NodeArg* left_input = node.MutableInputDefs()[0];
    auto left = IsInputTranspose(graph, *left_input);

    NodeArg* right_input = node.MutableInputDefs()[1];
    auto right = IsInputTranspose(graph, *right_input);

    if (!left.first && !right.first) {
      continue;
    }

    if (left.first) {
      size_t left_consumers = UpdateConsumerCount(graph, left_input, consumer_count);
      if (left_consumers == 0)
        removed_nodes.push_front(left.second->Index());
      left_input = left.second->MutableInputDefs()[0];
    }

    if (right.first) {
      size_t right_consumers = UpdateConsumerCount(graph, right_input, consumer_count);
      if (right_consumers == 0)
        removed_nodes.push_front(right.second->Index());
      right_input = right.second->MutableInputDefs()[0];
    }

    const std::vector<NodeArg*> input_defs{left_input, right_input};
    const std::vector<NodeArg*> output_defs{node.MutableOutputDefs()[0]};

    Node& matmul_node = graph.AddNode(graph.GenerateNodeName("MatMul_With_Transpose"),
                                      "TransposeMatMul",
                                      "fused MatMul and Transpose ",
                                      input_defs,
                                      output_defs, {}, kMSDomain);
    bool transpose_left = left.first;
    if (node.OpType() == "TransposeMatMul") {
      transpose_left ^= static_cast<bool>(node.GetAttributes().at("transA").i());
    }
    bool transpose_right = right.first;
    if (node.OpType() == "TransposeMatMul") {
      transpose_right ^= static_cast<bool>(node.GetAttributes().at("transB").i());
    }
    matmul_node.AddAttribute("transA", (int64_t)transpose_left);
    matmul_node.AddAttribute("transB", (int64_t)transpose_right);
    // Assign provider to this new node. Provider should be same as the provider for old node.
    matmul_node.SetExecutionProviderType(node.GetExecutionProviderType());

    graph_utils::FinalizeNodeFusion(graph, matmul_node, node);
  }

  // Have to remove node in reversed order for now to walk around the issue in RemoveNode
  for (onnxruntime::NodeIndex removed_node : removed_nodes) {
    graph.RemoveNode(removed_node);
  }

  if (!removed_nodes.empty()) {
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
