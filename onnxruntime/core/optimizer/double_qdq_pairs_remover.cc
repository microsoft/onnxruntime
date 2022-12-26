// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "double_qdq_pairs_remover.h"
#include "core/graph/graph_utils.h"
namespace onnxruntime {
Status DoubleQDQPairsRemover::ApplyImpl(
    Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // node was removed as part of an earlier optimization

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }
  return Status::OK();
}

  bool DoubleQDQPairsRemover::IsPairDQQRemovable(const Node &dq_node, const Node &q_node) const {
  auto dq_proto = dq_node.InputDefs()[1]->ToProto();
  auto q_proto = q_node.OutputDefs()[1]->ToProto();
  dq_proto.
  return true;
  }

  std::vector<const Node *> DoubleQDQPairsRemover::GetQualifiedQChildren(const Node &dq_node) const {
    auto q_children = graph_utils::FindChildrenByType(dq_node, "QuantizeLinear");
    q_children.erase(
        std::remove_if(
            q_children.begin(),
            q_children.end(),
            [&]( const Node* q_node) {
              return IsPairDQQRemovable(dq_node, *q_node);
            }),
        q_children.end());
    return q_children;
  }

}  // namespace onnxruntime
