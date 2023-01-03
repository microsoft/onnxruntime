// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/identical_children_consolidation.h"

#include "core/graph/graph_utils.h"

namespace onnxruntime {
Status IdenticalChildrenConsolidation::ApplyImpl(Graph& graph,
                                                 bool& modified,
                                                 int /*graph_level*/,
                                                 const logging::Logger& /*logger*/) const {
  GraphViewer const graph_viewer(graph);
  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    Node* node = graph.GetNode(node_index);
    if (IsSupportedParentNode(node)) {
      auto identical_children_indexes = GetIdenticalChildrenIndexes(node);
      if (identical_children_indexes.empty()) {
        continue;
      }
      auto* first_child = graph.GetNode(identical_children_indexes[0]);
      for (size_t i = 1; i < identical_children_indexes.size(); i++) {
        auto* other_child = graph.GetNode(identical_children_indexes[i]);
        // Currently Only support single output
        graph_utils::ReplaceDownstreamNodeInput(graph, *other_child, 0, *first_child, 0);
        graph_utils::RemoveNode(graph, *other_child);
        modified = true;
      }
    }
  }
  return Status::OK();
}

std::vector<NodeIndex> IdenticalChildrenConsolidation::GetIdenticalChildrenIndexes(Node* node) const {
  unordered_set<NodeIndex> identical_set;
  for (auto i = node->OutputEdgesBegin(); i != std::prev(node->OutputEdgesEnd(), 1); ++i) {
    if (supported_children_optypes.count(i->GetNode().OpType()) > 0) {
      for (auto j = std::next(i); j != node->OutputEdgesEnd(); ++j) {
        if (i->GetNode().OpType() == j->GetNode().OpType()) {
          identical_set.insert({i->GetNode().Index(), j->GetNode().Index()});
        }
      }
    }
  }
  return {identical_set.begin(), identical_set.end()};
}

bool IdenticalChildrenConsolidation::IsSupportedParentNode(const Node* node) const {
  return node != nullptr && supported_parent_optypes.count(node->OpType()) != 0 && node->GetOutputEdgesCount() > 1;
}
}  // namespace onnxruntime