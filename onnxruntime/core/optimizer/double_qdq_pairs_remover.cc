// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "double_qdq_pairs_remover.h"
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
}  // namespace onnxruntime
