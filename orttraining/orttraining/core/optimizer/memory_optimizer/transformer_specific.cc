// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <charconv>
#include <vector>
#include <utility>

#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/tensorprotoutils.h"

#include "core/common/string_utils.h"

namespace onnxruntime::optimizer::memory_optimizer {

void FindLayerBoundaryLayerNodeNodes(
    const GraphViewer& graph_viewer,
    const logging::Logger&,
    InlinedHashSet<const Node*>& layer_boundary_ln_nodes) {
  // Loop all nodes to find LayerNormalization nodes.
  // For each LayerNormalization node, keep checking its output nodes,
  // until find a node that is Softmax or BiasSoftmax or another LayerNormalization.
  // If the found node is Softmax or BiasSoftmax, the LayerNormalization node as ATTENTION.
  // If the found node is another LayerNormalization, the LayerNormalization node as MLP.
  const InlinedHashSet<std::string_view> softmax_ops{"Softmax", "BiasSoftmax"};
  const InlinedHashSet<std::string_view> layernorm_ops{"LayerNormalization", "SkipLayerNormalization"};

  layer_boundary_ln_nodes.clear();
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto& node = *graph_viewer.GetNode(node_index);

    if (layernorm_ops.find(node.OpType()) == layernorm_ops.end()) {
      continue;
    }

    std::deque<const Node*> nodes_to_check;
    std::set<const Node*> visited_nodes;
    for (auto node_it = node.OutputNodesBegin(); node_it != node.OutputNodesEnd(); ++node_it) {
      nodes_to_check.push_back(&(*node_it));
    }

    while (!nodes_to_check.empty()) {
      const Node* next_node = nodes_to_check.front();
      nodes_to_check.pop_front();

      if (visited_nodes.find(next_node) != visited_nodes.end()) {
        continue;
      }

      visited_nodes.insert(next_node);
      if (softmax_ops.find(next_node->OpType()) != softmax_ops.end()) {
        layer_boundary_ln_nodes.insert(&node);
        break;
      } else if (layernorm_ops.find(next_node->OpType()) != layernorm_ops.end()) {
        break;
      } else {
        for (auto node_it = next_node->OutputNodesBegin(); node_it != next_node->OutputNodesEnd(); ++node_it) {
          nodes_to_check.push_back(&(*node_it));
        }
      }
    }
  }
}

}  // namespace onnxruntime::optimizer::memory_optimizer
