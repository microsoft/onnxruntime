// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/transformer_layer_recompute.h"

namespace onnxruntime {

std::vector<NodeArg*> TransformerLayerRecompute::IdentifyTransformerLayerEdges(Graph& graph) const {
  std::vector<NodeArg*> layer_edges;

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);

    if (node.OpType() == "LayerNormalization") {
      layer_edges.push_back(node.InputDefs()[0]);
    }
  }
  return layer_edges;
}

std::vector<Node*> NodesBetweenEdges(Graph& graph, const NodeArg* begin, const NodeArg* end) const {
    

}

}  // namespace onnxruntime
