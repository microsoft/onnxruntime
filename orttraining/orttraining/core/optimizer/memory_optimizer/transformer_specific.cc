// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <charconv>
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

void FindLayerBoundaryLayerNormNodes(
    const GraphViewer& graph_viewer,
    const logging::Logger&,
    const InlinedHashMap<NodeIndex, ptrdiff_t>&
        node_index_to_its_order_in_topological_sort_map,
    const ptrdiff_t& yield_op_order_in_topological_sort,
    InlinedHashSet<const Node*>& layer_boundary_ln_nodes) {
  // Loop all nodes to find LayerNormalization nodes.
  // For each LayerNormalization node, keep checking its output nodes,
  // until find a node that is Softmax or BiasSoftmax or another LayerNormalization.
  // If the found node is Softmax or BiasSoftmax, the LayerNormalization node as ATTENTION.
  // If the found node is another LayerNormalization, the LayerNormalization node as MLP.
  const InlinedHashSet<std::string_view> softmax_ops{"Softmax", "BiasSoftmax"};
  const InlinedHashSet<std::string_view> layernorm_ops{"LayerNormalization", "SkipLayerNormalization"};

  layer_boundary_ln_nodes.clear();
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);
  for (auto node_index : node_topology_list) {
    auto& node = *graph_viewer.GetNode(node_index);

    if (layernorm_ops.find(node.OpType()) == layernorm_ops.end()) {
      continue;
    }

    std::deque<const Node*> nodes_to_check;
    std::set<const Node*> visited_nodes;
    for (auto node_it = node.OutputNodesBegin(); node_it != node.OutputNodesEnd(); ++node_it) {
      // Ignore those nodes after YieldOp.
      if (node_index_to_its_order_in_topological_sort_map.at(node_it->Index()) < yield_op_order_in_topological_sort) {
        nodes_to_check.push_back(&(*node_it));
      }
    }

    bool unexpected_failure = false;
    bool found_softmax = false;
    bool found_layernorm = false;
    ptrdiff_t next_layernorm_execution_oder = -1;
    while (!nodes_to_check.empty()) {
      const Node* next_node = nodes_to_check.front();
      nodes_to_check.pop_front();

      if (visited_nodes.find(next_node) != visited_nodes.end()) {
        continue;
      }

      visited_nodes.insert(next_node);
      if (softmax_ops.find(next_node->OpType()) != softmax_ops.end()) {
        found_softmax = true;
      } else if (layernorm_ops.find(next_node->OpType()) != layernorm_ops.end()) {
        if (found_layernorm) {
          // If we found another LayerNormalization node, we would report as warning, and do nothing for layer boundary detection.
          unexpected_failure = true;
          break;
        }
        found_layernorm = true;  // don't trace further
        next_layernorm_execution_oder = node_index_to_its_order_in_topological_sort_map.at(next_node->Index());
        continue;
      } else {
        for (auto node_it = next_node->OutputNodesBegin(); node_it != next_node->OutputNodesEnd(); ++node_it) {
          // Stop if the node is after next Layernorm node in execution order.
          if (found_layernorm &&
              node_index_to_its_order_in_topological_sort_map.at(node_it->Index()) >= next_layernorm_execution_oder) {
            continue;
          }
          nodes_to_check.push_back(&(*node_it));
        }
      }
    }

    if (unexpected_failure) {
      layer_boundary_ln_nodes.clear();
      break;
    }

    if (found_softmax) {
      layer_boundary_ln_nodes.insert(&node);
    } else if (!found_layernorm) {
      // If no Softmax found, and no other LayerNormalization found, this should be the last LayerNormalization node,
      // we also consider it as boundary node.
      layer_boundary_ln_nodes.insert(&node);
    }
  }
}

}  // namespace onnxruntime::optimizer::memory_optimizer
