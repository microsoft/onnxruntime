// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/quantized_embed_layer_norm_fusion.h"

namespace onnxruntime {

Status QuantizedEmbedLayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);

  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* p_layer_norm = graph.GetNode(node_index);
    if (p_layer_norm == nullptr) {
      // This node was removed as part of an earlier fusion?
      continue;
    }

    std::string op_type = p_layer_norm->OpType();
    if (op_type.size() > 0) {
    }

    //
    // TODO(kreeger): LEFT OFF RIGHT HERE. NEED TO WRITE THIS FUNCTION!
    //

    Node& layer_norm_node = *p_layer_norm;
    ORT_RETURN_IF_ERROR(Recurse(layer_norm_node, modified, graph_level, logger));
  }

  return Status::OK();
}

}  // namespace onnxruntime
