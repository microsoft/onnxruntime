// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/embed_layer_norm_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "float.h"

#define DEBUG_LOG(x) LOGS(logger, VERBOSE) << x

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// Add a Cast to convert Input from int64 to int32.
static NodeArg* CastToInt32(Graph& graph, NodeArg* input, ProviderType provider_type) {
  auto data_type = input->TypeAsProto()->tensor_type().elem_type();
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    return input;
  }
  const TensorShapeProto* input_shape = input->Shape();
  TypeProto input_int32;
  input_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  auto dim0 = input_int32.mutable_tensor_type()->mutable_shape()->add_dim();
  *dim0 = input_shape->dim(0);
  auto dim1 = input_int32.mutable_tensor_type()->mutable_shape()->add_dim();
  *dim1 = input_shape->dim(1);
  auto& cast32 = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(input->Name() + "_Int32"), &input_int32);

  Node& node = graph.AddNode(graph.GenerateNodeName(input->Name() + "_Cast"),
                             "Cast",
                             "Cast Input from int64 to int32",
                             {input},
                             {&cast32},
                             nullptr,
                             kOnnxDomain);

  // Add attribute: "to" = 6
  ONNX_NAMESPACE::AttributeProto to;
  to.set_name("to");
  to.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  to.set_i(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32));
  node.AddAttribute("to", to);

  node.SetExecutionProviderType(provider_type);
  return &cast32;
}

static bool CheckInput(NodeArg* input, const logging::Logger& logger) {
  // Validate input shape (batch_size, sequence_length) and data type.
  // Note that batch_size and sequence_length could be symbolic.
  const TensorShapeProto* input_shape = input->Shape();
  if (input_shape == nullptr || input_shape->dim_size() != 2 || input->Type() == nullptr) {
    DEBUG_LOG("Input shape is unknown or not 2D, or data type unknown");
    return false;
  }

  auto data_type = input->TypeAsProto()->tensor_type().elem_type();
  if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    DEBUG_LOG("Input data type is not int32 or int64");
    return false;
  }
  return true;
}

static bool MatchPositionEmbeddingSubgraph1(
    Graph& graph,
    Node& position_gather_node,
    NodeArg* input_ids,
    const logging::Logger& logger,
    std::vector<const Node::EdgeEnd*>& matched_edges) {
  // Match two paths.
  // Match Shape --> Expand path if needed.
  std::vector<NodeIndex> position_parent_nodes;
  std::vector<graph_utils::EdgeEndToMatch> position_embedding_path_symbolic{
      {0, 1, "Expand", {8}, kOnnxDomain},
      {0, 1, "Shape", {1}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(position_gather_node, true, position_embedding_path_symbolic, edges, logger)) {
    return false;
  }
  if (edges[0]->GetNode().GetOutputEdgesCount() != 1 && edges[1]->GetNode().GetOutputEdgesCount() != 1) {
    return false;
  }
  auto p_expand_node = graph.GetNode(edges[0]->GetNode().Index());
  auto p_shape_node = graph.GetNode(edges[1]->GetNode().Index());
  // Match Shape --> Gather --> Unsqueeze --> ConstantOfShape --> NonZero --> Transpose --> Squeeze --> Cast --> Unsqueeze --> Expand
  Node& expand_node = *graph.GetNode(edges[0]->GetNode().Index());
  Node& shape_node_1 = *graph.GetNode(edges[1]->GetNode().Index());
  std::vector<graph_utils::EdgeEndToMatch> pg_parent_path{
      {0, 0, "Unsqueeze", {1, 11}, kOnnxDomain},
      {0, 0, "Cast", {9}, kOnnxDomain},
      {0, 0, "Squeeze", {1}, kOnnxDomain},
      {0, 0, "Transpose", {1}, kOnnxDomain},
      {0, 0, "NonZero", {9}, kOnnxDomain},
      {0, 0, "ConstantOfShape", {9}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11}, kOnnxDomain},
      {0, 0, "Gather", {1, 11}, kOnnxDomain},
      {0, 0, "Shape", {1}, kOnnxDomain},
  };
  matched_edges = edges;

  if (!graph_utils::FindPath(expand_node, true, pg_parent_path, edges, logger)) {
    return false;
  }
  for (size_t i = 0; i < edges.size(); i++) {
    if (edges[i]->GetNode().GetOutputEdgesCount() != 1) {
      return false;
    }
  }
  // Check if the second input of the Gather node in the path has a constant input of 1
  Node& gather_node = *graph.GetNode(edges[edges.size() - 2]->GetNode().Index());
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather_node.InputDefs()[1]), int64_t(1), true)) {
    DEBUG_LOG("Second input of Gather should be a constant with value 1. ");
    return false;
  }

  // Check if the parent of "shape" is the input_ids
  Node& shape_node_2 = *graph.GetNode(edges[edges.size() - 1]->GetNode().Index());
  if (shape_node_1.MutableInputDefs()[0] != input_ids ||
      shape_node_1.MutableInputDefs()[0] != input_ids) {
    return false;
  }

  matched_edges.insert(matched_edges.end(), edges.begin(), edges.end());
  return true;
}

/** Match subgraph like the following:
            (input_ids)
          /             \
     Shape               Shape
       |                    |
    Gather (indice=0)    Gather (indice=1)--+
       |                    |               |
    Unsqueeze            Unsqueeze         Cast
         \             /                    |
          \           /                Range(start=0, delta=1)
           \         /                      |
             Concat                       Unsqueeze
               |                            |
            +--|----------------------------+
            |  |
           Expand
              |
            Gather

 Note that position gather node is the node in the bottom of above sub-graph.
*/

static bool MatchPositionEmbeddingSubgraph2(
    Graph& graph,
    Node& position_gather_node,
    NodeArg* input_ids,
    const logging::Logger& logger,
    std::vector<const Node::EdgeEnd*>& matched_edges) {

  // Match Gather <-- Expand <-- Unsqueeze <-- Range <-- Cast <-- Gather <-- Shape
  std::vector<NodeIndex> position_parent_nodes;
  std::vector<graph_utils::EdgeEndToMatch> position_embedding_path_symbolic{
      {0, 1, "Expand", {8}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11}, kOnnxDomain},
      {0, 0, "Range", {11}, kOnnxDomain},
      {0, 1, "Cast", {9}, kOnnxDomain},
      {0, 0, "Gather", {11}, kOnnxDomain},
      {0, 0, "Shape", {1}, kOnnxDomain},
  };
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(position_gather_node, true, position_embedding_path_symbolic, edges, logger)) {
    DEBUG_LOG("Failed to find path 1.");
    return false;
  }
  for (size_t i = 0; i < edges.size(); i++) {
    if (edges[i]->GetNode().GetOutputEdgesCount() != (i == 4 ? 2 : 1)) {
      DEBUG_LOG("Output edge count not expected for nodes in path 1.");
      return false;
    }
  }
  matched_edges = edges;

  Node& expand_node = *graph.GetNode(edges[0]->GetNode().Index());
  Node& range_node = *graph.GetNode(edges[2]->GetNode().Index());
  Node& gather_node_1 = *graph.GetNode(edges[4]->GetNode().Index());
  Node& shape_node_1 = *graph.GetNode(edges[5]->GetNode().Index());
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(range_node.InputDefs()[0]), int64_t(0), true)) {
    DEBUG_LOG("The first input of Range should be a constant with value 0.");
    return false;
  }
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(range_node.InputDefs()[2]), int64_t(1), true)) {
    DEBUG_LOG("The third input of Range should be a constant with value 1.");
    return false;
  }
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather_node_1.InputDefs()[1]), int64_t(1), true)) {
    DEBUG_LOG("The second input of Gather in path1 should be a constant with value 1.");
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> expand_parent_path1{
      {0, 1, "Concat", {11}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11}, kOnnxDomain},
      {0, 0, "Gather", {1, 11}, kOnnxDomain},
      {0, 0, "Shape", {1}, kOnnxDomain},
  };
  if (!graph_utils::FindPath(expand_node, true, expand_parent_path1, edges, logger)) {
    DEBUG_LOG("Failed to find path 2.");
    return false;
  }
  for (size_t i = 0; i < edges.size(); i++) {
    if (edges[i]->GetNode().GetOutputEdgesCount() != 1) {
      DEBUG_LOG("Output edge count not expected for nodes in path 2.");
      return false;
    }
  }

  Node& concat_node = *graph.GetNode(edges[0]->GetNode().Index());
  Node& gather_node_0 = *graph.GetNode(edges[2]->GetNode().Index());
  Node& shape_node_0 = *graph.GetNode(edges[3]->GetNode().Index());
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather_node_0.InputDefs()[1]), int64_t(0), true)) {
    DEBUG_LOG("Second input of Gather in path2 should be a constant with value 0.");
    return false;
  }
  matched_edges.insert(matched_edges.end(), edges.begin(), edges.end());

  std::vector<graph_utils::EdgeEndToMatch> concat_parent_path{
      {0, 1, "Unsqueeze", {1, 11}, kOnnxDomain},
      {0, 0, "Gather", {1, 11}, kOnnxDomain}
  };
  if (!graph_utils::FindPath(concat_node, true, concat_parent_path, edges, logger)) {
    DEBUG_LOG("Failed to find path 3.");
    return false;
  }
  // Two paths share the gather node (with second input indices==1)
  if (edges[1]->GetNode().Index() != gather_node_1.Index()) {
    DEBUG_LOG(" Gather nodes in path 1 and 3 expected to be same node.");
    return false;
  }
  if (edges[0]->GetNode().GetOutputEdgesCount() != 1) {
    DEBUG_LOG("Output edge count not expected for nodes in path 3.");
    return false;
  }

  // Check if the two paths of position gather lead to the same input.
  if (shape_node_0.MutableInputDefs()[0] != input_ids ||
      shape_node_1.MutableInputDefs()[0] != input_ids) {
    DEBUG_LOG("Two shape nodes are expected to be input_ids.");
    return false;
  }

  // Do not add the gather node since it has been added in another path.
  matched_edges.push_back(edges[0]);
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
    auto* p_layer_norm = graph.GetNode(node_index);
    if (p_layer_norm == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& layer_norm_node = *p_layer_norm;
    ORT_RETURN_IF_ERROR(Recurse(layer_norm_node, modified, graph_level, logger));
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(layer_norm_node, "LayerNormalization", {1}, kOnnxDomain) ||
        !graph_utils::IsSupportedProvider(layer_norm_node, GetCompatibleExecutionProviders())) {
      continue;
    }
    // Find Attention after SkipLayerNormalization
    const Node* p_attention = graph_utils::FirstChildByType(layer_norm_node, "Attention");
    // Stop EmbedLayerNormalization fusion if Attention is not found.
    if (p_attention == nullptr) {
      return Status::OK();
    }
    Node& attention_node = *graph.GetNode(p_attention->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(attention_node, "Attention", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(attention_node, GetCompatibleExecutionProviders())) {
      continue;
    }
    // Find ReduceSum --> Attention
    std::vector<const Node::EdgeEnd*> edges;
    if (!graph_utils::FindPath(attention_node, true, {{0, 3, "ReduceSum", {1, 11}, kOnnxDomain}}, edges, logger)) {
      continue;
    }
    Node& reduce_sum_node = *graph.GetNode(edges[0]->GetNode().Index());

    // Find Add --> LayerNormalization
    if (!graph_utils::FindPath(layer_norm_node, true, {{0, 0, "Add", {7}, kOnnxDomain}}, edges, logger)) {
      continue;
    }
    Node& layer_norm_add_node = *graph.GetNode(edges[0]->GetNode().Index());

    // Traceback the SkipLayerNormalization node to find Gather --> SkipLayerNormalization
    std::vector<graph_utils::EdgeEndToMatch> segment_embedding_path{
        {0, 1, "Gather", {1, 11}, kOnnxDomain}};
    if (!graph_utils::FindPath(layer_norm_add_node, true, segment_embedding_path, edges, logger)) {
      continue;
    }
    Node& segment_gather_node = *graph.GetNode(edges[0]->GetNode().Index());
    if (segment_gather_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // The first input of segment_gather_node must be 2d.
    NodeArg* segment_embedding = segment_gather_node.MutableInputDefs()[0];
    auto sg_shape = segment_embedding->Shape();
    if (sg_shape == nullptr || sg_shape->dim_size() != 2 ||
        !utils::HasDimValue(sg_shape->dim()[1]) ||
        sg_shape->dim()[1].dim_value() <= 0) {
      continue;
    }

    // Traceback the SkipLayerNormalization node to find Gather --> Add --> SkipLayerNormalization
    std::vector<graph_utils::EdgeEndToMatch> word_embedding_path{
        {0, 0, "Add", {7}, kOnnxDomain},
        {0, 0, "Gather", {1, 11}, kOnnxDomain}};
    if (!graph_utils::FindPath(layer_norm_add_node, true, word_embedding_path, edges, logger)) {
      continue;
    }
    Node& add_node = *graph.GetNode(edges[0]->GetNode().Index());
    Node& word_gather_node = *graph.GetNode(edges[1]->GetNode().Index());
    if (add_node.GetOutputEdgesCount() != 1 || word_gather_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // The first input of word_gather_node must be 2d.
    NodeArg* word_embedding = word_gather_node.MutableInputDefs()[0];
    auto wg_shape = word_embedding->Shape();
    if (wg_shape == nullptr || wg_shape->dim_size() != 2 ||
        !utils::HasDimValue(wg_shape->dim()[1]) ||
        wg_shape->dim()[1].dim_value() != sg_shape->dim()[1].dim_value()) {
      continue;
    }

    // Traceback the Add node to find (Shape --> Expand -->) Gather --> Add.
    // Constant folding removes Shape and Expand nodes when input does not have symbolic shape. In that
    // case just look for Gather --> Add.
    std::vector<graph_utils::EdgeEndToMatch> position_embedding_path{
        {0, 1, "Gather", {1, 11}, kOnnxDomain}};
    if (!graph_utils::FindPath(add_node, true, position_embedding_path, edges, logger)) {
      continue;
    }
    Node& position_gather_node = *graph.GetNode(edges[0]->GetNode().Index());
    if (position_gather_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // The first input of position_gather_node must be 2d.
    NodeArg* position_embedding = position_gather_node.MutableInputDefs()[0];
    auto pg_shape = position_embedding->Shape();
    if (pg_shape == nullptr || pg_shape->dim_size() != 2 ||
        !utils::HasDimValue(pg_shape->dim()[1]) ||
        pg_shape->dim()[1].dim_value() != sg_shape->dim()[1].dim_value()) {
      continue;
    }

    NodeArg* input_ids = word_gather_node.MutableInputDefs()[1];

    // Check the second input of position gather. If it's not initializer, check for two paths.
    std::vector<const Node::EdgeEnd*> pg_edges;
    if (graph_utils::IsConstantInitializer(graph, position_gather_node.MutableInputDefs()[1]->Name())) {
      // Check if the second input of position gather is a tensor with values evenly spaced by 1 starting from 0.
      std::vector<int64_t> data;
      auto expected_shape = word_gather_node.MutableInputDefs()[1]->Shape();
      if (!optimizer_utils::AppendTensorFromInitializer(graph, *(position_gather_node.MutableInputDefs()[1]), data) || !utils::HasDimValue(expected_shape->dim()[0]) || !utils::HasDimValue(expected_shape->dim()[1]) || static_cast<int>(data.size()) != expected_shape->dim()[0].dim_value() * expected_shape->dim()[1].dim_value()) {
        continue;
      }
      int64_t expected_value = 0;
      bool isValidEmbedSubNode = true;
      for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != expected_value) {
          isValidEmbedSubNode = false;
          break;
        }
        expected_value++;
        if (expected_value >= static_cast<int64_t>(expected_shape->dim()[1].dim_value())) {
          expected_value = 0;
        }
      }
      if (!isValidEmbedSubNode) {
        continue;
      }
    } else {
      if (!MatchPositionEmbeddingSubgraph1(graph, position_gather_node, input_ids, logger, pg_edges)) {
        pg_edges.clear();
        if (!MatchPositionEmbeddingSubgraph2(graph, position_gather_node, input_ids, logger, pg_edges)) {
          continue;
        }
      }
    }

    // Get input "input_ids" from node.
    if (!CheckInput(input_ids, logger)) {
      DEBUG_LOG("Input id is not valid. ");
      continue;
    }

    // Get input "segment_ids" from node.
    NodeArg* segment_ids = segment_gather_node.MutableInputDefs()[1];
    if (!CheckInput(segment_ids, logger)) {
      DEBUG_LOG("Segment id is not valid. ");
      continue;
    }

    // Get input "mask" from "ReduceSum" node.
    NodeArg* mask = reduce_sum_node.MutableInputDefs()[0];
    if (!CheckInput(mask, logger)) {
      DEBUG_LOG("Mask is not valid. ");
      continue;
    }

    if (utils::GetTensorShapeFromTensorShapeProto(*(input_ids->Shape())) !=
        utils::GetTensorShapeFromTensorShapeProto(*(segment_ids->Shape()))) {
      DEBUG_LOG("Input_ids and segment id should have the same shape. ");
      continue;
    }
    if (utils::GetTensorShapeFromTensorShapeProto(*(input_ids->Shape())) !=
        utils::GetTensorShapeFromTensorShapeProto(*(mask->Shape()))) {
      DEBUG_LOG("Input_ids and mask should have the same shape. ");
      continue;
    }

    NodeArg* gamma = layer_norm_node.MutableInputDefs()[1];
    NodeArg* beta = layer_norm_node.MutableInputDefs()[2];
    if (gamma->Shape() == nullptr || gamma->Shape()->dim()[0].dim_value() != word_embedding->Shape()->dim()[1].dim_value()) {
      DEBUG_LOG("Gamma should be of shape (hidden_size). ");
      continue;
    }

    if (beta->Shape() == nullptr || beta->Shape()->dim()[0].dim_value() != word_embedding->Shape()->dim()[1].dim_value()) {
      DEBUG_LOG("Beta should be of shape (hidden_size). ");
      continue;
    }

    // Cast input_ids, segment_ids, and mask to int32 if needed.
    input_ids = CastToInt32(graph, input_ids, layer_norm_node.GetExecutionProviderType());
    segment_ids = CastToInt32(graph, segment_ids, layer_norm_node.GetExecutionProviderType());
    mask = CastToInt32(graph, mask, layer_norm_node.GetExecutionProviderType());

    const std::vector<NodeArg*> embed_layer_norm_input_defs{
        input_ids,
        segment_ids,
        word_embedding,
        position_embedding,
        segment_embedding,
        layer_norm_node.MutableInputDefs()[1],
        layer_norm_node.MutableInputDefs()[2],
        mask};
    Node& embed_layer_norm_node = graph.AddNode(graph.GenerateNodeName("EmbedLayerNormalization"),
                                                "EmbedLayerNormalization",
                                                "fused EmbedLayerNorm subgraphs ",
                                                embed_layer_norm_input_defs,
                                                {layer_norm_node.MutableOutputDefs()[0], reduce_sum_node.MutableOutputDefs()[0]},
                                                {}, kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    embed_layer_norm_node.SetExecutionProviderType(layer_norm_node.GetExecutionProviderType());

    // move input edges to gather (first in list) across to the embed_layer_norm_node.
    // move output definitions and output edges to embed_layer_norm_node.
    // remove all the other nodes.
    std::vector<NodeIndex> nodes_to_remove;
    for (size_t i = 0; i < pg_edges.size(); i++) {
      nodes_to_remove.push_back(pg_edges[i]->GetNode().Index());
    }
    nodes_to_remove.push_back(word_gather_node.Index());
    nodes_to_remove.push_back(position_gather_node.Index());
    nodes_to_remove.push_back(segment_gather_node.Index());
    nodes_to_remove.push_back(add_node.Index());
    nodes_to_remove.push_back(reduce_sum_node.Index());
    nodes_to_remove.push_back(layer_norm_add_node.Index());
    nodes_to_remove.push_back(layer_norm_node.Index());

    for (const auto& index : nodes_to_remove) {
      Node* node = graph.GetNode(index);
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
    }
    modified = true;
  }
  return Status::OK();
}
}  // namespace onnxruntime