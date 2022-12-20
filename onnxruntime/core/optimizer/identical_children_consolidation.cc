// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/graph/graph_utils.h>
#include "identical_children_consolidation.h"

namespace onnxruntime {
Status IdenticalChildrenConsolidation::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  for (auto node_index : graph_viewer.GetRootNodes()) {
    Node& node = *(graph.GetNode(node_index));
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }
  for (auto node_index : graph_viewer.GetRootNodes()) {
    Node& node = *(graph.GetNode(node_index));
    // 1. Checking if the node is a supported op type.
    if (supported_ops_and_supported_children.find(node.OpType()) != supported_ops_and_supported_children.end()) {
      // 2. Grouping the children nodes by their OpType.
      unordered_map<std::string_view,std::vector<Node*>> children_groups;
      const auto& supported_children_ops = supported_ops_and_supported_children.at(node.OpType());
      const auto& group_size = supported_children_ops.size();
      // 3. Checking if the node has children that are supported op types and if the child has single parent.
      // If yes, add them to the children vector.
      for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
        if (supported_children_ops.find(it->GetNode().OpType()) != supported_children_ops.end()
            && it->GetNode().GetInputEdgesCount() == 1) {
          children_groups[it->GetNode().OpType()].emplace_back(&it->GetNode());
        }
      }
      // 4. Checking if each group of the children vector has more than one node.
      for (const auto& children_group: children_groups){
          const auto& children = children_group.second;
          std::vector<NodeArg*> input_defs = node.MutableInputDefs();
          if (children.size() > 1) {
            // 5. Consolidating the children nodes.
            Node& new_node = graph.AddNode(
                graph.GenerateNodeName(node.Name() /*+ "_consolidated"*/),
                node.OpType(),
                node.Description(),
                node.MutableInputDefs(),
                node.MutableOutputDefs(),// todo: check if this is correct
                &node.GetAttributes(),
                node.Domain()
            );
            for (auto& child : children) {
              graph_utils::RemoveNodeOutputEdges(graph, *child);
              graph.RemoveNode(child->Index());
            }
//            graph_utils::FinalizeNodeFusion(graph, new_node, children); todo: check if this is correct
            modified = true;
          }
      }
    }
  }
  return Status::OK();
}
}