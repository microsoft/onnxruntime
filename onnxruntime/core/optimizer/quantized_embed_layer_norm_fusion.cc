// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/quantized_embed_layer_norm_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

namespace {

// NOTE: These are not in the proper order. I wonder if there is a way to loop
// through all edges of a node and do it that way.
constexpr int kEmbedLayerNormWordDequantizeInputIdx = 2;
constexpr int kEmbedLayerNormPositionDequantizeInputIdx = 3;
constexpr int kEmbedLayerNormSegmentDequantizeInputIdx = 4;  // Also known as "token"

//void GetNodesFromEmbedLayerNormalizationNode(const Node& node) {
//  //
//  // TODO - write me.
//  //
//}

}  // namespace

Status QuantizedEmbedLayerNormFusion::ApplyImpl(Graph& graph,
                                                bool& modified,
                                                int graph_level,
                                                const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);

  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node = graph.GetNode(node_index);
    if (node == nullptr) {
      // TODO - validate that the node was removed as part of an earlier fusion...
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    // 1.) Find the "EmbedLayerNormalization" node:
    if (node->OpType().compare("EmbedLayerNormalization") != 0 ||
        !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // Try and find the correct input thing here for a node?
    auto inputs = node->InputDefs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto* input_arg = inputs.at(i);
      printf("  - input_arg: %s\n", input_arg->Name().c_str());
    }

    // GetInputNode TODO - use this API.

    // Find and assing the 
    //Node* p_word_dequantize_node = nullptr;
    //Node* p_position_dequantize_node = nullptr;
    //Node* p_segment_dequantize_node = nullptr;

    // Get the parent nodes of the dequantizelinear

    // 2.) Find the "DequantizeLinear" Node for the position embedding:
    // 3.) Find the "DequantizeLinear" Node for the segment embedding:
    // 4.) Find the "DequantizeLinear" Node for the word embedding:
  }

  return Status::OK();
}

Node* QuantizedEmbedLayerNormFusion::FindNodeFromPath(const Node& node,
                                                      Graph& graph,
                                                      const char* op_type,
                                                      const logging::Logger& logger) const {

  //
  // TODO(kreeger): Left off right here. Need to determine why this method is not finding
  // any of the DequantizeLinear operators in the graph. Might have to look at other APIs 
  // to examine the usage of graph_utils::FindPath()...
  //

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(node,
                             /*is_input_edge=*/true,
                             {{/*src_arg_index=*/0,
                               /*dst_arg_index=*/0,
                               op_type,
                               /*version=*/{10},
                               kMSDomain}},
                             edges,
                             logger)) {
    return nullptr;
  }

  return graph.GetNode(edges[0]->GetNode().Index());

}

}  // namespace onnxruntime
