// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "float.h"
#include <algorithm>
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status LoraWeightsFolding::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    nodes_to_remove.clear();

    // Look for the self attention path that looks like the following:
    //
    //                  <AnyNode>
    //          ____________|_____________
    //         |      |         |         |
    //         |    MatMul    MatMul    MatMul
    //         |      |         |         |
    //         |    MatMul    MatMul    MatMul
    //         |      |         |         |
    //         |     Mul       Mul       Mul
    //         |      |         |         |
    //         |     Mul       Mul       Mul
    //         |      |         |         |
    //         |   Reshape   Reshape   Reshape
    //         |      |_________|_________|
    //         |                |
    //         |              Concat
    //         |                |
    //       MatMul          Reshape
    //         \_____   _______/
    //               \ /
    //               Add
    //
    // Essentially, we are able to fold all MatMul nodes to the right of the Add node into the left node
    // by adding their initializers together, as long as there's a node (called AnyNode here) that feeds
    // into their A input, and as long as their B inputs are initializers hardcoded into the model.
    auto* add_node = graph.GetNode(node_index);
    if (add_node == nullptr || add_node->OpType() != "Add") {
      continue;
    }

    std::vector<graph_utils::EdgeEndToMatch> add_left_path{
        {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
    };

    std::vector<std::reference_wrapper<Node>> add_left_path_nodes;
    if (!graph_utils::FindPath(*add_node, true, add_left_path, add_left_path_nodes, logger)) {
      continue;
    }

    std::vector<graph_utils::EdgeEndToMatch> add_right_path{
        {1, 0, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
        {0, 0, "Concat", {1, 4, 11, 13}, kOnnxDomain},
    };

    std::vector<std::reference_wrapper<Node>> add_right_path_nodes;
    if (!graph_utils::FindPath(*add_node, true, add_right_path, add_right_path_nodes, logger)) {
      continue;
    }

    const Node& concat_node = add_right_path_nodes[1];

    if (concat_node.InputDefs().size() != 3) {
      continue;
    }

    std::vector<graph_utils::EdgeEndToMatch> concat_q_path{
        {0, 0, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
        {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
        {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
        {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
        {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
    };

    std::vector<std::reference_wrapper<Node>> concat_q_path_nodes;
    if (!graph_utils::FindPath(concat_node, true, concat_q_path, concat_q_path_nodes, logger)) {
      return false;
    }

    std::vector<graph_utils::EdgeEndToMatch> concat_k_path{
        {1, 0, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
        {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
        {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
        {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
        {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
    };

    std::vector<std::reference_wrapper<Node>> concat_k_path_nodes;
    if (!graph_utils::FindPath(concat_node, true, concat_k_path, concat_k_path_nodes, logger)) {
      return false;
    }

    std::vector<graph_utils::EdgeEndToMatch> concat_v_path{
        {2, 0, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
        {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
        {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
        {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
        {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
    };

    std::vector<std::reference_wrapper<Node>> concat_v_path_nodes;
    if (!graph_utils::FindPath(concat_node, true, concat_v_path, concat_v_path_nodes, logger)) {
      return false;
    }

    const auto& qkv_matmul_node = add_left_path_nodes[0];

    const auto& q_scale_node = concat_q_path_nodes[1];
    const auto& q_alpha_node = concat_q_path_nodes[2];
    const auto& q_lora_up_node = concat_q_path_nodes[3];
    const auto& q_lora_down_node = concat_q_path_nodes[4];

    const auto& k_scale_node = concat_k_path_nodes[1];
    const auto& k_alpha_node = concat_k_path_nodes[2];
    const auto& k_lora_up_node = concat_k_path_nodes[3];
    const auto& k_lora_down_node = concat_k_path_nodes[4];

    const auto& v_scale_node = concat_v_path_nodes[1];
    const auto& v_alpha_node = concat_v_path_nodes[2];
    const auto& v_lora_up_node = concat_v_path_nodes[3];
    const auto& v_lora_down_node = concat_v_path_nodes[4];

    std::array<std::reference_wrapper<Node>, 13> all_nodes = {
        qkv_matmul_node,
        q_scale_node,
        q_alpha_node,
        q_lora_up_node,
        q_lora_down_node,
        k_scale_node,
        k_alpha_node,
        k_lora_up_node,
        k_lora_down_node,
        v_scale_node,
        v_alpha_node,
        v_lora_up_node,
        v_lora_down_node,
    };

    // Make sure that the B input of each MatMul and Mul node is a constant initializer (i.e. an initializer
    // that cannot be overridden by an input at runtime).
    const bool have_constant_weights = !std::all_of(all_nodes.begin(), all_nodes.end(), [](auto node) {
      auto input_defs = node.InputDefs();

      if (input_defs.size() != 2) {
        return false;
      }

      const ONNX_NAMESPACE::TensorProto* initializer = GetConstantInitializer(graph, input_def->Name(), true);
      if (initializer == nullptr) {
        return false;
      }

      return true;
    });

    if (!have_constant_weights) {
      continue;
    }

    // Make sure that the root MatMul nodes all share the same input
    std::array<std::reference_wrapper<Node>, 4> root_matmul_nodes = {
        qkv_matmul_node,
        q_lora_down_node,
        k_lora_down_node,
        v_lora_down_node,
    };

    const bool share_same_root = std::equal(root_matmul_nodes.begin(), root_matmul_nodes.end(), [](auto first_node, auto second_node) {
      auto first_node_input_defs = first_node.InputDefs();
      if (first_node_input_defs.size() != 2) {
        return false;
      }

      auto second_node_input_defs = second_node.InputDefs();
      if (second_node_input_defs.size() != 2) {
        return false;
      }

      if (first_node_input_defs[0]->Index() != second_node_input_defs[0]->Index()) {
        return false;
      }

      return true;
    });

    if (!share_same_root) {
      continue;
    }

    // We need to compute the matmul operation on the CPU. What's the best way to do this???

    std::array<std::reference_wrapper<Node>, 3> lora_nodes = {
        q_lora_down_node,
        k_lora_down_node,
        v_lora_down_node,
    };

    modified = true;
  }
  return Status::OK();
}

}  // namespace onnxruntime
