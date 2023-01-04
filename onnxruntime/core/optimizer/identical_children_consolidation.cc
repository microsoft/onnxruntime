// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/identical_children_consolidation.h"

#include "core/graph/graph_utils.h"

namespace onnxruntime {
Status IdenticalChildrenConsolidation::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& /*logger*/) const {
  GraphViewer const graph_viewer(graph);
  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    Node* node = graph.GetNode(node_index);
    if (!IsSupportedParentNode(node)) {
      continue;
    }
    for (auto supported_op : supported_ops.at(node->OpType())) {
      for (auto twin_group : DivideIdenticalChildrenIntoGroups(graph, node, supported_op)) {
        // If there is no twins in the group, skip it.
        if (twin_group.size() <= 1) {
          continue;
        }
        Node* first_twin = graph.GetNode(twin_group[0]);
        for (size_t i = 1; i < twin_group.size(); i++) {
          Node* other_twin = graph.GetNode(twin_group[i]);
          // Replace all outputs of other_twin with first_twin
          for (size_t j = 0; j < other_twin->GetOutputEdgesCount(); j++) {
            graph_utils::ReplaceDownstreamNodeInput(graph, *other_twin, static_cast<int>(j), *first_twin, static_cast<int>(j));
          }
          graph_utils::RemoveNode(graph, *other_twin);
          modified = true;
        }
      }
    }
  }
  return Status::OK();
}

bool IdenticalChildrenConsolidation::IsSupportedParentNode(const Node* node) const {
  return node != nullptr && supported_ops.count(node->OpType()) != 0 && node->GetOutputEdgesCount() > 1;
}

std::vector<std::vector<NodeIndex>> IdenticalChildrenConsolidation::DivideIdenticalChildrenIntoGroups(const Graph& graph, Node* node, const string_view& op) const {
  unordered_map<string_view, std::vector<NodeIndex>> identical_children_map;
  for (auto i = node->OutputEdgesBegin(); i != node->OutputEdgesEnd(); ++i) {
    if (i->GetNode().OpType() == op) {
      identical_children_map[IdentityBuilder(graph, i->GetNode())].push_back(i->GetNode().Index());
    }
  }
  std::vector<std::vector<NodeIndex>> groups;
  for (auto& identical_children : identical_children_map) {
    groups.push_back(identical_children.second);
  }
  return groups;
}

string_view IdenticalChildrenConsolidation::IdentityBuilder(const Graph& graph, const Node& node) const {
  std::string identity;
  for (const auto* input_def : node.InputDefs()) {
    if (input_def->Exists() && !input_def->Name().empty()) {
      auto name = input_def->Name();
      if (graph_utils::NodeArgIsConstant(graph, *input_def)) {
        identity.append(constant_prefix).append(graph_utils::GetConstantInitializer(graph, name)->raw_data());
      } else {
        identity.append(name.substr(name.find_last_of('/') + 1));
      }
    }
  }
  return {identity};
}
}  // namespace onnxruntime