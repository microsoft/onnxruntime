// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/embed_layer_norm_fusion.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// EmbedLayerNorm supports limited data types.
static std::vector<std::string> supported_data_types{"tensor(float16)", "tensor(float)", "tensor(int32)"};

static bool IsSupportedDataType(const Node& node) {
  for (const auto& input_arg : node.InputDefs()) {
    if (std::find(supported_data_types.begin(), supported_data_types.end(),
                  *(input_arg->Type())) == supported_data_types.end()) {
      return false;
    }
  }
  return true;
}

/**
Embed Layer Normalization will fuse embeddings and mask processing into one node : 
The embeddings before conversion:
  (input_ids) -------->  Gather ----------+       (segment_ids)
    |                                    |            |
    |                                    v            v
    +--> Shape --> Expand -> Gather---->Add         Gather
    |                ^                   |            |
    |                |                   v            v
    +---(optional graph)               SkipLayerNormalization

*/
Status EmbedLayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* p_skip_ln = graph.GetNode(node_index);
    if (p_skip_ln == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& skip_ln_node = *p_skip_ln;
    ORT_RETURN_IF_ERROR(Recurse(skip_ln_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(skip_ln_node, "SkipLayerNormalization", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(skip_ln_node, GetCompatibleExecutionProviders()) ||
        !IsSupportedDataType(skip_ln_node)) {
      continue;
    }
    // Find Attention after SkipLayerNormalization
    const Node* p_attention = graph_utils::FirstChildByType(skip_ln_node, "Attention");
    if (p_attention == nullptr) {
      continue;
    }
    Node& attention_node = *graph.GetNode(p_attention->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(attention_node, "Attention", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(attention_node, GetCompatibleExecutionProviders()) ||
        !IsSupportedDataType(attention_node)) {
      continue;
    }
    // Find ReduceSum --> Attention
    std::vector<const Node::EdgeEnd*> edges;
    if (!graph_utils::FindPath(attention_node, true, {{0, 3, "ReduceSum", {1}, kOnnxDomain}}, edges, logger)) {
      continue;
    }
    Node& reduce_sum_node = *graph.GetNode(edges[0]->GetNode().Index());
    if (reduce_sum_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    // Traceback the SkipLayerNormalization node to find Gather --> SkipLayerNormalization
    std::vector<graph_utils::EdgeEndToMatch> segment_embedding_path{
        {0, 0, "Gather", {1}, kOnnxDomain}};
    edges.clear();
    if (!graph_utils::FindPath(skip_ln_node, true, segment_embedding_path, edges, logger)) {
      continue;
    }
    Node& segment_gather_node = *graph.GetNode(edges[0]->GetNode().Index());
    if (segment_gather_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    // Traceback the SkipLayerNormalization node to find Gather --> Add --> SkipLayerNormalization
    std::vector<graph_utils::EdgeEndToMatch> word_embedding_path{
        {0, 1, "Add", {7}, kOnnxDomain},
        {0, 0, "Gather", {1}, kOnnxDomain}};
    edges.clear();
    if (!graph_utils::FindPath(skip_ln_node, true, word_embedding_path, edges, logger)) {
      continue;
    }
    Node& add_node = *graph.GetNode(edges[0]->GetNode().Index());
    Node& word_gather_node = *graph.GetNode(edges[1]->GetNode().Index());
    if (add_node.GetOutputEdgesCount() != 1 || word_gather_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    // Traceback the Add node to find Shape --> Expand --> Gather --> Add
    std::vector<graph_utils::EdgeEndToMatch> position_embedding_path{
        {0, 1, "Gather", {1}, kOnnxDomain}};
        //{0, 1, "Expand", {8}, kOnnxDomain},
        //{0, 1, "Shape", {1}, kOnnxDomain}};
    edges.clear();
    if (!graph_utils::FindPath(add_node, true, position_embedding_path, edges, logger)) {
      continue;
    }
    Node& position_gather_node = *graph.GetNode(edges[0]->GetNode().Index());
    //Node& expand_node = *graph.GetNode(edges[1]->GetNode().Index());
    //Node& shape_node = *graph.GetNode(edges[2]->GetNode().Index());
    if (position_gather_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    // Get input "input_ids" from node.
    NodeArg* input_ids = nullptr;
    if (!graph_utils::NodeArgIsConstant(graph, *(word_gather_node.MutableInputDefs()[1])) &&
        !graph_utils::IsGraphInput(graph, word_gather_node.MutableInputDefs()[1])) {
      continue;
    }
    input_ids = word_gather_node.MutableInputDefs()[1];

    // Get input "segment_ids" from node.
    NodeArg* segment_ids = nullptr;
    if (!graph_utils::NodeArgIsConstant(graph, *(segment_gather_node.MutableInputDefs()[1])) &&
        !graph_utils::IsGraphInput(graph, segment_gather_node.MutableInputDefs()[1])) {
      continue;
    }
    segment_ids = segment_gather_node.MutableInputDefs()[1];

    // Get input "mask" from "ReduceSum" node.
    NodeArg* mask = nullptr;
    if (!graph_utils::NodeArgIsConstant(graph, *(reduce_sum_node.MutableInputDefs()[0])) &&
        !graph_utils::IsGraphInput(graph, reduce_sum_node.MutableInputDefs()[0])) {
      continue;
    }
    mask = reduce_sum_node.MutableInputDefs()[0];

    const std::vector<NodeArg*> embed_layer_norm_input_defs{
        input_ids,
        segment_ids,
        mask,
        word_gather_node.MutableInputDefs()[0],
        position_gather_node.MutableInputDefs()[0],
        segment_gather_node.MutableInputDefs()[0],
        skip_ln_node.MutableInputDefs()[2],
        skip_ln_node.MutableInputDefs()[3]};
    Node& embed_layer_norm_node = graph.AddNode(graph.GenerateNodeName("EmbedLayerNormalization"),
                                                "EmbedLayerNormalization", 
                                                "fused EmbedLayerNorm subgraphs ", 
                                                embed_layer_norm_input_defs,
                                                {skip_ln_node.MutableOutputDefs()[0], reduce_sum_node.MutableOutputDefs()[0]},
                                                {}, kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    embed_layer_norm_node.SetExecutionProviderType(skip_ln_node.GetExecutionProviderType());

    // move input edges to add (first in list) across to the embed_layer_norm_node.
    // move output definitions and output edges to embed_layer_norm_node.
    // remove all the other nodes.
    std::vector<Node*> nodes_to_remove{
        graph.GetNode(word_gather_node.Index()),
        graph.GetNode(position_gather_node.Index()),
        //graph.GetNode(shape_node.Index()),
        //graph.GetNode(expand_node.Index()),
        graph.GetNode(segment_gather_node.Index()),
        graph.GetNode(add_node.Index()),
        graph.GetNode(reduce_sum_node.Index()),
        graph.GetNode(skip_ln_node.Index())
    };

    for (auto* node : nodes_to_remove) {
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
    }

    modified = true;
  }
  return Status::OK();
}
}  // namespace onnxruntime