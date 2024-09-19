// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <charconv>
#include <tuple>
#include <vector>
#include <utility>

#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/transformer_specific.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/tensorprotoutils.h"

#include "core/common/string_utils.h"

namespace onnxruntime::optimizer::memory_optimizer {

namespace {

bool IsLayerNormNode(const Node& node) {
  static const std::set<std::string> layer_norm_ops = {
      "LayerNormalization",
      "SkipLayerNormalization",
      "SimplifiedLayerNormalization",
      "SkipSimplifiedLayerNormalization",
  };
  return layer_norm_ops.find(node.OpType()) != layer_norm_ops.end();
}

bool IsSoftmaxNode(const Node& node) {
  static const std::set<std::string> softmax_ops = {
      "Softmax",
      "BiasSoftmax",
  };
  return softmax_ops.find(node.OpType()) != softmax_ops.end();
}

bool IsAttentionOp(const Node& node) {
  if (node.OpType() != "PythonOp") {
    return false;
  }

  // Check the func_name attribute of the PythonOp node.
  const auto* func_name_attr = graph_utils::GetNodeAttribute(node, "func_name");
  if (func_name_attr == nullptr) {
    return false;
  }

  static const std::set<std::string> attn_op_names = {
      "flash_attn.flash_attn_interface.FlashAttnVarlenFunc",
      "flash_attn.flash_attn_interface.FlashAttnFunc",
  };
  return attn_op_names.find(func_name_attr->s()) != attn_op_names.end();
}

std::tuple<bool, const Node*, const Node*> IsResidualNodeArg(const GraphViewer& graph_viewer, const NodeArg* node_arg) {
  auto consumers = graph_viewer.GetConsumerNodes(node_arg->Name());
  if (2 > consumers.size()) {
    return std::make_tuple(false, nullptr, nullptr);
  }

  // Find the Add node from the consumer list.
  const Node* add_node = nullptr;
  const Node* other_node = nullptr;
  for (const auto* consumer : consumers) {
    if (consumer->OpType() == "Add") {
      add_node = consumer;
    } else if (IsLayerNormNode(*consumer)) {
      other_node = consumer;
    }
  }

  return std::make_tuple(add_node != nullptr && other_node != nullptr, add_node, other_node);
}
}  // namespace

/*
    One classical layer includes 1). input layer norm, 2). attention, 3). residual add
    (input layer norm input + attention output), 4). post attention layer norm feedforward, and 5). residual add
    (post attention layer norm input + feedforward out).

    The pattern graph looks like below for each transformer layer (taking the example of MistralDecoderLayer):
                            |
                        Embedding
                            |
      ----------------------|
      |                     |
      |                     |
      |        SimplifiedLayerNormalization (layer boundary node)
      |                     |
      |                     |
      |               MistralAttention
      |                     |
      |                     |
      |____________________Add
                            |
      ----------------------|
      |                     |
      |                     |
      |         SimplifiedLayerNormalization
      |                     |
      |                     |
      |            MultipleLayerPerception
      |                     |
      |                     |
      |____________________Add
                            |
                        (new layer)
      ----------------------|
      |                     |
      |         SimplifiedLayerNormalization
                           ....
*/
void FindLayerBoundaryLayerNormNodes(
    const GraphViewer& graph_viewer,
    const logging::Logger& logger,
    const InlinedHashMap<NodeIndex, ptrdiff_t>&
        node_index_to_its_order_in_topological_sort_map,
    const ptrdiff_t& yield_op_order_in_topological_sort,
    InlinedVector<const Node*>& layer_boundary_ln_nodes) {
  // Loop all nodes to find LayerNormalization nodes.
  // For each LayerNormalization node, keep checking its output nodes,
  // until find a node that is Softmax or Attention or another LayerNormalization.
  // If the found node is Softmax or Attention, the LayerNormalization node as ATTENTION.
  // If the found node is another LayerNormalization, the LayerNormalization node as MLP.

  layer_boundary_ln_nodes.clear();

  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder(TOPOLOGICAL_SORT_ALGORITHM);
  for (auto node_index : node_topology_list) {
    auto& node = *graph_viewer.GetNode(node_index);

    if (!IsLayerNormNode(node)) {
      continue;
    }
    const NodeArg* input_arg = node.InputDefs()[0];

    // IsResidualNodeArg checks input_arg
    auto [is_residual_node_arg, add_node, other_node] = IsResidualNodeArg(graph_viewer, input_arg);
    if (!is_residual_node_arg) {
      MO_LOG_DEBUG_INFO(logger, "Not a residual node arg " + input_arg->Name());
      continue;
    }

    // At this point, there should not be any recompute node, so we don't need check the node existence in
    //  node_index_to_its_order_in_topological_sort_map.
    ptrdiff_t attention_residual_add_node_order =
        node_index_to_its_order_in_topological_sort_map.at(add_node->Index());
    ptrdiff_t attention_residual_ln_node_order =
        node_index_to_its_order_in_topological_sort_map.at(other_node->Index());
    if (attention_residual_add_node_order >= yield_op_order_in_topological_sort ||
        attention_residual_ln_node_order >= yield_op_order_in_topological_sort) {
      MO_LOG_DEBUG_INFO(logger, "Not a valid residual node arg " + input_arg->Name());
      continue;
    }

    // Search all forward nodes that is before `add_node` in topo order, unless we find a softmax or
    // nodes_to_check is empty.
    std::deque<const Node*> nodes_to_check;
    std::set<const Node*> visited_nodes;
    for (auto node_it = node.OutputNodesBegin(); node_it != node.OutputNodesEnd(); ++node_it) {
      // Ignore those nodes after YieldOp.
      auto order = node_index_to_its_order_in_topological_sort_map.at(node_it->Index());
      if (order < yield_op_order_in_topological_sort && order < attention_residual_add_node_order) {
        nodes_to_check.push_back(&(*node_it));
      }
    }

    while (!nodes_to_check.empty()) {
      const Node* next_node = nodes_to_check.front();
      nodes_to_check.pop_front();

      if (visited_nodes.find(next_node) != visited_nodes.end()) {
        continue;
      }

      visited_nodes.insert(next_node);
      if (IsSoftmaxNode(*next_node) || IsAttentionOp(*next_node)) {
        MO_LOG_DEBUG_INFO(logger, "Found layer boundary node " + node.Name() + " with its input arg: " +
                                      input_arg->Name());
        layer_boundary_ln_nodes.push_back(&node);
        break;
      }

      for (auto node_it = next_node->OutputNodesBegin(); node_it != next_node->OutputNodesEnd(); ++node_it) {
        // Stop if the node is after next Layernorm node in execution order.
        auto order = node_index_to_its_order_in_topological_sort_map.at(node_it->Index());
        if (order < yield_op_order_in_topological_sort && order < attention_residual_add_node_order) {
          nodes_to_check.push_back(&(*node_it));
        }
      }
    }
  }
}

}  // namespace onnxruntime::optimizer::memory_optimizer
