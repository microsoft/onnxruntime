// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/embed_layer_norm_fusion.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "float.h"
#include "core/common/safeint.h"

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

static bool IsNeighborNodeExpectedTypes(Node::NodeConstIterator start, const Node::NodeConstIterator end, const std::vector<std::string>& expected_types) {
  for (const std::string& expected_type : expected_types) {
    if (start == end || (*start).OpType().compare(expected_type) != 0) {
      return false;
    }
    ++start;
  }
  return start == end;
}

/** Match subgraph like the following:
            (input_ids)
          /             \
     Shape               Shape
       |                    |
    Gather (indice=0)    Gather (indice=1)--+
       |                    |               |
    Unsqueeze            Unsqueeze          |
         \             /                    |
          \           /                  [other subgraph]
           \         /                      |
             Concat                         |
               |                            |
               +----------------------------+--+
                                            |  |
                                          [Expand]
                                              |
                                          [Gather] (for position embedding)

 Note that expand_node is the Expand node in the graph, and
 expected_gather_node_1_index is node index of the gather with indices=1.

 The Expand and Gather on the bottom will not be added to subgraph_node_indices.
 It is because they are matched as part of other subgraph.

 Two Shape nodes may merge into one.
*/

static bool MatchInputToConcatSubgraph(
    Graph& graph,
    const Node& cur_node,
    const NodeArg* input_ids,
    const int index,
    const logging::Logger& logger,
    const NodeIndex expected_gather_node_1_index) {
  std::vector<graph_utils::EdgeEndToMatch> expand_parent_path1{
      {0, index, "Concat", {4, 11, 13}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain},
  };

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(cur_node, true, expand_parent_path1, edges, logger)) {
    DEBUG_LOG("Failed to find path 1 of position shape.");
    return false;
  }
  const size_t shape_index = edges.size() - 1;
  for (size_t i = 0; i < edges.size(); i++) {
    if (!optimizer_utils::CheckOutputEdges(graph, edges[i]->GetNode(), 1)) {
      // Shape may have multiple outputs due to shape integration
      // So check it later
      if (i == shape_index) {
        continue;
      }
      DEBUG_LOG("Output edge count not expected for nodes in path 1 of position shape.");
      return false;
    }
  }

  Node& concat_node = *graph.GetNode(edges[0]->GetNode().Index());
  Node& gather_node_0 = *graph.GetNode(edges[2]->GetNode().Index());
  Node& shape_node_0 = *graph.GetNode(edges[3]->GetNode().Index());
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather_node_0.InputDefs()[1]), int64_t(0), true)) {
    DEBUG_LOG("Second input of Gather in path 1 of position shape should be a constant with value 0.");
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> concat_parent_path{
      {0, 1, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};

  if (!graph_utils::FindPath(concat_node, true, concat_parent_path, edges, logger)) {
    DEBUG_LOG("Failed to find path 2 of position shape.");
    return false;
  }

  // Shape may have multiple outputs due to shape integration
  // Check it later
  if (!optimizer_utils::CheckOutputEdges(graph, edges[0]->GetNode(), 1) ||
      !optimizer_utils::CheckOutputEdges(graph, edges[1]->GetNode(), 2)) {
    DEBUG_LOG("Output edge count not expected for nodes in path 2 of position shape.");
    return false;
  }

  Node& gather_node_1 = *graph.GetNode(edges[1]->GetNode().Index());
  Node& shape_node_1 = *graph.GetNode(edges[2]->GetNode().Index());

  // The gather node (with second input indices==1) is also shared by other subgraph
  if (expected_gather_node_1_index != gather_node_1.Index()) {
    DEBUG_LOG("Gather node in path 2 is not linked to another subgraph.");
    return false;
  }

  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather_node_1.InputDefs()[1]), int64_t(1), true)) {
    DEBUG_LOG("Second input of Gather in path 2 of position shape should be a constant with value 1.");
    return false;
  }

  // Check if the two paths of position gather lead to the same input.
  if (shape_node_0.MutableInputDefs()[0] != input_ids ||
      shape_node_1.MutableInputDefs()[0] != input_ids) {
    DEBUG_LOG("The parent of two shape nodes are expected to be input_ids.");
    return false;
  }

  // Check if shape have more than one output, it may due to shape integration
  // We check if they share the same node
  if (!optimizer_utils::CheckOutputEdges(graph, shape_node_0, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, shape_node_1, 1)) {
    if (shape_node_0.Index() == shape_node_1.Index() &&
        (shape_node_0.GetOutputEdgesCount() == 2 ||
         shape_node_0.GetOutputEdgesCount() == 4)) {
      DEBUG_LOG("two paths share the same shape");
    } else {
      return false;
    }
  }

  return true;
}

/** Match subgraph like the following:
 *
 *    Shape -> ^Gather (indice=0)^ -> ^Unsqueeze^
 *      /                                  |           +-----------------------+
 *     /                                   v           |                       |
 * [input_ids]                          ^Concat^ -> *Reshape* -> *Equal* -> *Where* -> Expand -> Gather
 *     \                                   |                                            |   ("position")
 *    Shape -> ^Gather (indice=1)^ -> ^Unsqueeze^                                       |
 *                |                                                                     |
 *                +-------------- # one of the below subgraph patterns # ---------------+
 *       # Unsqueeze -> ConstantOfShape -> NonZero -> Transpose -> Squeeze -> (Cast) -> Unsqueeze #
 *       #                                      or                                                #
 *       #              (Cast (to=7)) -> Range (start=0, delta=1) -> Unsqueeze                    #
 *
 * Note that position gather node is the node in the bottom of above sub-graph.
 * Paths in ^^ are alternative path to be matched if path input_ids -> Shape -> Expand -> Gather is not found.
 * Path in ** is an alternative path to check.
 * Two shape node may merge into one
 */
static bool MatchPositionEmbeddingSubgraphsFromGather(
    Graph& graph,
    const Node& position_gather_node,
    const NodeArg* input_ids,
    const logging::Logger& logger) {
  std::vector<const Node::EdgeEnd*> pg_edges;
  // Look for Path 1:
  // Shape --> Gather --> Unsqueeze --> ConstantOfShape --> NonZero --> Transpose --> Squeeze
  // --> Cast --> Unsqueeze --> Expand --> Gather
  std::vector<graph_utils::EdgeEndToMatch> parent_path_1{
      {0, 1, "Expand", {8, 13}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Cast", {9, 13}, kOnnxDomain},
      {0, 0, "Squeeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "NonZero", {9, 13}, kOnnxDomain},
      {0, 0, "ConstantOfShape", {9}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};
  // Look for Path 2 (Path 1 with no cast):
  std::vector<graph_utils::EdgeEndToMatch> parent_path_2{
      {0, 1, "Expand", {8, 13}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Squeeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "NonZero", {9, 13}, kOnnxDomain},
      {0, 0, "ConstantOfShape", {9}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};
  // Path 3 Pattern:
  // Shape -> Gather -> Cast (to=7) -> Range (start=0, delta=1) -> Unsqueeze -> Expand
  std::vector<graph_utils::EdgeEndToMatch> parent_path_3{
      {0, 1, "Expand", {8, 13}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Range", {1, 11}, kOnnxDomain},
      {0, 1, "Cast", {9, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};
  // Path 4 pattern (Path 3 with no "Cast"):
  std::vector<graph_utils::EdgeEndToMatch> parent_path_4{
      {0, 1, "Expand", {8, 13}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Range", {1, 11}, kOnnxDomain},
      {0, 1, "Gather", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};
  // Match one of the three path patterns.
  if (!graph_utils::FindPath(position_gather_node, true, parent_path_1, pg_edges, logger) &&
      !graph_utils::FindPath(position_gather_node, true, parent_path_2, pg_edges, logger) &&
      !graph_utils::FindPath(position_gather_node, true, parent_path_3, pg_edges, logger) &&
      !graph_utils::FindPath(position_gather_node, true, parent_path_4, pg_edges, logger)) {
    return false;
  }
  const size_t gather_index = pg_edges.size() - 2;
  const size_t shape_index = pg_edges.size() - 1;
  // All nodes in Path 1 must have only 1 output edge, except the gather node allowed 1 or 2 output edges
  // And shape node allowed multiple output edges due to shape integration
  for (size_t i = 0; i < pg_edges.size(); i++) {
    if (!optimizer_utils::CheckOutputEdges(graph, pg_edges[i]->GetNode(), 1)) {
      if (i == gather_index && optimizer_utils::CheckOutputEdges(graph, pg_edges[i]->GetNode(), 2)) {
        continue;
      }
      if (i == shape_index &&
          (optimizer_utils::CheckOutputEdges(graph, pg_edges[i]->GetNode(), 2) ||
           optimizer_utils::CheckOutputEdges(graph, pg_edges[i]->GetNode(), 4))) {
        continue;
      }
      DEBUG_LOG("Output edge count not expected for nodes in path1.");
      return false;
    }
  }

  Node& expand_node = *graph.GetNode(pg_edges[0]->GetNode().Index());
  Node& gather_node = *graph.GetNode(pg_edges[gather_index]->GetNode().Index());
  if (pg_edges[2]->GetNode().OpType() == "Range") {
    // Check if the values in "start" and "delta" attributes in Range are expected.
    Node& range_node = *graph.GetNode(pg_edges[2]->GetNode().Index());
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(range_node.InputDefs()[0]), int64_t(0), true)) {
      DEBUG_LOG("The first input of Range should be a constant with value 0.");
      return false;
    }
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(range_node.InputDefs()[2]), int64_t(1), true)) {
      DEBUG_LOG("The third input of Range should be a constant with value 1.");
      return false;
    }
  }

  if (gather_node.GetOutputEdgesCount() == 1) {
    // Check if the second input of the Gather node in the path has a constant input of 1
    // For gather_output_edges_count == 2, such checks are in MatchPositionSubgraph function.
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather_node.InputDefs()[1]), int64_t(1), true)) {
      DEBUG_LOG("Second input of Gather should be a constant with value 1. ");
      return false;
    }

    // Match Shape --> Expand path.
    std::vector<const Node::EdgeEnd*> pg_edges_2;
    if (!graph_utils::FindPath(expand_node, true, {{0, 1, "Shape", {1, 13}, kOnnxDomain}}, pg_edges_2, logger)) {
      DEBUG_LOG("Failed to match Shape node. ");
      return false;
    }
    auto shape_node_index = pg_edges_2[0]->GetNode().Index();

    // Check if the two paths of position gather lead to the same input.
    Node& shape_node_1 = *graph.GetNode(pg_edges[pg_edges.size() - 1]->GetNode().Index());
    Node& shape_node_2 = *graph.GetNode(shape_node_index);
    if (shape_node_1.MutableInputDefs()[0] != input_ids ||
        shape_node_2.MutableInputDefs()[0] != input_ids) {
      DEBUG_LOG("The parent of shape nodes are expected to be input_ids.");
      return false;
    }
  } else {  // gather_output_edges_count == 2
    // Match optional Reshape -> Equal -> Where -> Expand
    //                  |                  |
    //                  --------------------
    std::vector<const Node::EdgeEnd*> pg_edges_2;
    std::vector<graph_utils::EdgeEndToMatch> path_to_match_1{
        {0, 1, "Where", {9}, kOnnxDomain},
        {0, 0, "Equal", {1, 7, 11, 13}, kOnnxDomain},
        {0, 0, "Reshape", {5, 13}, kOnnxDomain}};
    if (graph_utils::FindPath(expand_node, true, path_to_match_1, pg_edges_2, logger)) {
      if (!optimizer_utils::CheckOutputEdges(graph, pg_edges_2[0]->GetNode(), 1) ||
          !optimizer_utils::CheckOutputEdges(graph, pg_edges_2[1]->GetNode(), 1) ||
          !optimizer_utils::CheckOutputEdges(graph, pg_edges_2[2]->GetNode(), 2)) {
        DEBUG_LOG("Optional position subgraph nodes number of outputs unexpected.");
        return false;
      }
      Node& where_node = *graph.GetNode(pg_edges_2[0]->GetNode().Index());
      Node& reshape_node = *graph.GetNode(pg_edges_2[2]->GetNode().Index());
      if (where_node.MutableInputDefs()[2] != reshape_node.MutableOutputDefs()[0]) {
        DEBUG_LOG("Optional position subgraph nodes Where node is expected to be the parent of Reshape.");
        return false;
      }
      // Match [input_ids] -> Gather -> Shape -> Unsqueeze from Reshape node.
      if (!MatchInputToConcatSubgraph(graph, reshape_node, input_ids, 0, logger, gather_node.Index())) {
        DEBUG_LOG("Failed to match position subgraph.");
        return false;
      }
    } else if (!MatchInputToConcatSubgraph(graph, expand_node, input_ids, 1, logger, gather_node.Index())) {
      // Match [input_ids] -> Gather -> Shape -> Unsqueeze from Expand node.
      DEBUG_LOG("Failed to match position subgraph.");
      return false;
    }
  }

  return true;
}

static bool MatchPositionEmbeddingSubgraph(
    Graph& graph,
    const Node& add_node,
    const NodeArg* input_ids,
    const logging::Logger& logger,
    std::vector<NodeIndex>& subgraph_node_indices,
    NodeArg*& position_embedding) {
  // Traceback the Add node to find (Shape --> Expand -->) Gather --> Add.
  // Constant folding removes Shape and Expand nodes when input has static shape.
  // In that case just look for Gather --> Add.
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(add_node, true, {{0, 1, "Gather", {1, 11, 13}, kOnnxDomain}}, edges, logger)) {
    return false;
  }
  Node& position_gather_node = *graph.GetNode(edges[0]->GetNode().Index());
  if (!optimizer_utils::CheckOutputEdges(graph, position_gather_node, 1)) {
    return false;
  }

  // The first input of position_gather_node must be 2d.
  position_embedding = position_gather_node.MutableInputDefs()[0];

  // Check the second input of position gather. There are the following cases:
  // (1) it is initializer.
  // (2) it is not initializer and matches subgraph 1 (for opset 10) or 2 (for opset 11).
  if (graph_utils::IsConstantInitializer(graph, position_gather_node.MutableInputDefs()[1]->Name())) {
    // Check that the tensor has shape (batch_size, sequence_length)
    std::vector<int64_t> data;
    auto expected_shape = input_ids->Shape();
    if (!optimizer_utils::AppendTensorFromInitializer(graph, *(position_gather_node.MutableInputDefs()[1]), data) ||
        !utils::HasDimValue(expected_shape->dim()[0]) ||
        !utils::HasDimValue(expected_shape->dim()[1]) ||
        static_cast<int>(data.size()) != expected_shape->dim()[0].dim_value() * expected_shape->dim()[1].dim_value()) {
      return false;
    }

    // Check the tensor value is like [0, 1, ..., sequence_length -1] for each batch.
    int64_t expected_value = 0;
    for (size_t i = 0; i < data.size(); i++) {
      if (data[i] != expected_value) {
        return false;
      }
      expected_value++;
      if (expected_value >= static_cast<int64_t>(expected_shape->dim()[1].dim_value())) {
        expected_value = 0;
      }
    }
  } else {
    if (!MatchPositionEmbeddingSubgraphsFromGather(graph, position_gather_node, input_ids, logger)) {
      return false;
    }
  }

  subgraph_node_indices.clear();
  subgraph_node_indices.push_back(position_gather_node.Index());
  return true;
}

template <typename T>
bool CheckEmbeddingData(const T* data, int64_t batch_size, int64_t element_count) {
  // check that all batches has same data.
  size_t data_length = SafeInt<size_t>(batch_size) * element_count;
  for (size_t i = gsl::narrow<size_t>(element_count); i < data_length; i++) {
    if (data[i] != data[i % element_count]) {
      return false;
    }
  }
  return true;
}

static NodeArg* ExtractEmbedding(Graph& graph,
                                 int64_t batch_size,
                                 int64_t sequence_length,
                                 int64_t hidden_size,
                                 const ONNX_NAMESPACE::TensorProto* tensor,
                                 bool& modified) {
  assert(nullptr != tensor);
  assert(batch_size > 0);
  assert(sequence_length > 0);
  assert(hidden_size > 0);

  Initializer old_initializer{*tensor, graph.ModelPath()};
  auto data_type = tensor->data_type();

  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name(graph.GenerateNodeArgName("position_embeddings"));
  initializer.add_dims(sequence_length);
  initializer.add_dims(hidden_size);
  initializer.set_data_type(data_type);
  const int64_t element_count = sequence_length * hidden_size;

  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const float* data = old_initializer.data<float>();
    if (!CheckEmbeddingData(data, batch_size, element_count)) {
      return nullptr;
    }

    initializer.set_raw_data(data, gsl::narrow<size_t>(element_count) * sizeof(float));
  } else {  // data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
    const MLFloat16* data = old_initializer.data<MLFloat16>();
    if (!CheckEmbeddingData(data, batch_size, element_count)) {
      return nullptr;
    }

    initializer.set_raw_data(data, gsl::narrow<size_t>(element_count) * sizeof(MLFloat16));
  }

  NodeArg& node_arg = graph_utils::AddInitializer(graph, initializer);
  modified = true;
  return &node_arg;
}

static void CreateEmbedLayernormNode(Graph& graph,
                                     NodeArg* input_ids,
                                     NodeArg* segment_ids,
                                     NodeArg* word_embedding,
                                     NodeArg* position_embedding,
                                     NodeArg* segment_embedding,
                                     Node& layer_norm_node) {
  // Cast input_ids and segment_ids to int32 if needed.
  input_ids = CastToInt32(graph, input_ids, layer_norm_node.GetExecutionProviderType());
  if (segment_ids != nullptr && segment_embedding != nullptr) {
    segment_ids = CastToInt32(graph, segment_ids, layer_norm_node.GetExecutionProviderType());
  }

  NodeArg place_holder("", nullptr);
  if (segment_ids == nullptr && segment_embedding == nullptr) {
    segment_ids = &place_holder;
    segment_embedding = &place_holder;
  }

  const std::vector<NodeArg*> embed_layer_norm_input_defs{
      input_ids,
      segment_ids,
      word_embedding,
      position_embedding,
      segment_embedding,
      layer_norm_node.MutableInputDefs()[1],
      layer_norm_node.MutableInputDefs()[2]};

  auto& mask_index = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("mask_index"), nullptr);

  Node& embed_layer_norm_node = graph.AddNode(graph.GenerateNodeName("EmbedLayerNormalization"),
                                              "EmbedLayerNormalization",
                                              "fused EmbedLayerNorm subgraphs ",
                                              embed_layer_norm_input_defs,
                                              {layer_norm_node.MutableOutputDefs()[0], &mask_index},
                                              {}, kMSDomain);

  // Get attribute "epsilon" from "LayerNormalization" node if available. Else, default value
  // will be used.
  NodeAttributes ln_attrs = layer_norm_node.GetAttributes();
  NodeAttributes::const_iterator epsilon = ln_attrs.find("epsilon");
  if (epsilon != ln_attrs.end()) {
    embed_layer_norm_node.AddAttribute("epsilon", epsilon->second);
  } else {
    embed_layer_norm_node.AddAttribute("epsilon", contrib::kDefaultEmbedLayerNormEpsilon);
  }

  // Assign provider to this new node. Provider should be same as the provider for old node.
  embed_layer_norm_node.SetExecutionProviderType(layer_norm_node.GetExecutionProviderType());
}

static bool FuseSubGraph(Graph& graph,
                         Node& layer_norm_add_node,
                         Node& layer_norm_node,

                         bool& modified,
                         const logging::Logger& logger) {
  // Trace back to find the Gather for segment embedding.
  std::vector<graph_utils::EdgeEndToMatch> segment_embedding_path{
      {0, 1, "Gather", {1, 11, 13}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(layer_norm_add_node, true, segment_embedding_path, edges, logger)) {
    return false;
  }
  Node& segment_gather_node = *graph.GetNode(edges[0]->GetNode().Index());
  if (!optimizer_utils::CheckOutputEdges(graph, segment_gather_node, 1)) {
    return false;
  }
  // The first input of segment_gather_node must be 2d.
  NodeArg* segment_embedding = segment_gather_node.MutableInputDefs()[0];
  auto sg_shape = segment_embedding->Shape();
  if (sg_shape == nullptr || sg_shape->dim_size() != 2 ||
      !utils::HasDimValue(sg_shape->dim()[1]) ||
      sg_shape->dim()[1].dim_value() <= 0) {
    return false;
  }
  auto hidden_size = sg_shape->dim()[1].dim_value();

  // Trace back to find Gather --> Add --> LayerNormalization
  std::vector<graph_utils::EdgeEndToMatch> word_embedding_path{
      {0, 0, "Add", {7, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain}};
  if (!graph_utils::FindPath(layer_norm_add_node, true, word_embedding_path, edges, logger)) {
    return false;
  }
  Node& add_node = *graph.GetNode(edges[0]->GetNode().Index());
  Node& word_gather_node = *graph.GetNode(edges[1]->GetNode().Index());
  if (!optimizer_utils::CheckOutputEdges(graph, add_node, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, word_gather_node, 1)) {
    return false;
  }
  // The first input of word_gather_node must be 2d.
  NodeArg* word_embedding = word_gather_node.MutableInputDefs()[0];
  auto wg_shape = word_embedding->Shape();
  if (wg_shape == nullptr || wg_shape->dim_size() != 2 ||
      !utils::HasDimValue(wg_shape->dim()[1]) ||
      wg_shape->dim()[1].dim_value() != hidden_size) {
    DEBUG_LOG("Word embedding shape not expected.");
    return false;
  }

  NodeArg* input_ids = word_gather_node.MutableInputDefs()[1];
  NodeArg* position_embedding = nullptr;
  std::vector<NodeIndex> nodes_to_remove;

  // ORT constant folding might be applied to position embedding subgraph when input has static shape.
  // Here we handle such special case that the input of add node is constant initializer.
  auto add_input_name = add_node.MutableInputDefs()[1]->Name();
  if (graph_utils::IsConstantInitializer(graph, add_input_name)) {
    // Check that input has static shape.
    auto input_shape = input_ids->Shape();
    if (input_shape->dim_size() != 2 ||
        !utils::HasDimValue(input_shape->dim()[0]) ||
        !utils::HasDimValue(input_shape->dim()[1])) {
      DEBUG_LOG("Input is expected to have dim value in all dimensions.");
      return false;
    }

    int64_t batch_size = input_shape->dim()[0].dim_value();
    int64_t sequence_length = input_shape->dim()[1].dim_value();
    if (batch_size <= 0 || sequence_length <= 0) {
      return false;
    }

    const ONNX_NAMESPACE::TensorProto* position_embed_tensor;
    if (!graph.GetInitializedTensor(add_input_name, position_embed_tensor)) {
      DEBUG_LOG("Failed to get initializer tensor.");
      return false;
    }
    // Tensor shape shall be [batch_size, sequence_length, hidden_size].
    if (position_embed_tensor->dims_size() != 3 ||
        position_embed_tensor->dims(0) != batch_size ||
        position_embed_tensor->dims(1) != sequence_length ||
        position_embed_tensor->dims(2) != hidden_size) {
      DEBUG_LOG("Position embedding shape not matched.");
      return false;
    }

    // Tensor data type should be float or float16.
    const auto data_type = position_embed_tensor->data_type();
    if (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      DEBUG_LOG("Position embedding data type shall be float or float16.");
      return false;
    }

    // The tensor has same data for all batches, and we extract only one batch data as position embedding.
    position_embedding = ExtractEmbedding(graph, batch_size, sequence_length, hidden_size, position_embed_tensor, modified);
  } else {
    if (!MatchPositionEmbeddingSubgraph(graph, add_node, input_ids, logger, nodes_to_remove, position_embedding)) {
      DEBUG_LOG("Failed to match position embedding subgraph.");
      return false;
    }
  }

  if (position_embedding == nullptr) {
    DEBUG_LOG("Failed to get position embedding weights.");
    return false;
  }

  auto pg_shape = position_embedding->Shape();
  if (pg_shape == nullptr || pg_shape->dim_size() != 2 ||
      !utils::HasDimValue(pg_shape->dim()[1]) ||
      pg_shape->dim()[1].dim_value() != hidden_size) {
    DEBUG_LOG("Position embedding shape is not expected.");
    return false;
  }

  // Get input "input_ids" from node.
  if (!CheckInput(input_ids, logger)) {
    DEBUG_LOG("Input id is not valid. ");
    return false;
  }

  // Get input "segment_ids" from node.
  NodeArg* segment_ids = segment_gather_node.MutableInputDefs()[1];
  if (!CheckInput(segment_ids, logger)) {
    DEBUG_LOG("Segment id is not valid. ");
    return false;
  }

  if (utils::GetTensorShapeFromTensorShapeProto(*(input_ids->Shape())) !=
      utils::GetTensorShapeFromTensorShapeProto(*(segment_ids->Shape()))) {
    DEBUG_LOG("Input_ids and segment id should have the same shape. ");
    return false;
  }

  NodeArg* gamma = layer_norm_node.MutableInputDefs()[1];
  NodeArg* beta = layer_norm_node.MutableInputDefs()[2];
  if (gamma->Shape() == nullptr || gamma->Shape()->dim()[0].dim_value() != hidden_size) {
    DEBUG_LOG("Gamma should be of shape (hidden_size). ");
    return false;
  }

  if (beta->Shape() == nullptr || beta->Shape()->dim()[0].dim_value() != hidden_size) {
    DEBUG_LOG("Beta should be of shape (hidden_size). ");
    return false;
  }

  CreateEmbedLayernormNode(graph, input_ids, segment_ids, word_embedding, position_embedding, segment_embedding,
                           layer_norm_node);

  if (!nodes_to_remove.empty()) {
    graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *graph.GetNode(nodes_to_remove[0]));
  }

  nodes_to_remove.clear();

  nodes_to_remove.push_back(word_gather_node.Index());
  nodes_to_remove.push_back(segment_gather_node.Index());
  nodes_to_remove.push_back(add_node.Index());

  nodes_to_remove.push_back(layer_norm_add_node.Index());
  nodes_to_remove.push_back(layer_norm_node.Index());

  for (const NodeIndex index : nodes_to_remove) {
    Node* node = graph.GetNode(index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  return true;
}

// DistilBert's pattern does not have segment embedding
static bool FuseSubGraphDistilBert(Graph& graph,
                                   Node& layer_norm_add_node,
                                   Node& layer_norm_node,
                                   const logging::Logger& logger) {
  std::vector<graph_utils::EdgeEndToMatch> word_embedding_path{
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(layer_norm_add_node, true, word_embedding_path, edges, logger)) {
    return false;
  }
  Node& word_gather_node = *graph.GetNode(edges[0]->GetNode().Index());
  if (!optimizer_utils::CheckOutputEdges(graph, word_gather_node, 1)) {
    return false;
  }
  // The first input of word_gather_node must be 2d.
  NodeArg* word_embedding = word_gather_node.MutableInputDefs()[0];
  auto wg_shape = word_embedding->Shape();
  if (wg_shape == nullptr || wg_shape->dim_size() != 2 ||
      !utils::HasDimValue(wg_shape->dim()[1])) {
    DEBUG_LOG("Word embedding shape not expected.");
    return false;
  }

  int64_t hidden_size = wg_shape->dim()[1].dim_value();

  Node& add_node = layer_norm_add_node;

  NodeArg* input_ids = word_gather_node.MutableInputDefs()[1];
  NodeArg* position_embedding = nullptr;
  std::vector<NodeIndex> nodes_to_remove;

  // ORT constant folding might be applied to position embedding subgraph when input has static shape.
  // Here we handle such special case that the input of add node is constant initializer.
  auto add_input_name = add_node.MutableInputDefs()[1]->Name();
  if (!MatchPositionEmbeddingSubgraph(graph, add_node, input_ids, logger, nodes_to_remove, position_embedding)) {
    DEBUG_LOG("Failed to match position embedding subgraph.");
    return false;
  }

  if (position_embedding == nullptr) {
    DEBUG_LOG("Failed to get position embedding weights.");
    return false;
  }

  auto pg_shape = position_embedding->Shape();
  if (pg_shape == nullptr || pg_shape->dim_size() != 2 ||
      !utils::HasDimValue(pg_shape->dim()[1]) ||
      pg_shape->dim()[1].dim_value() != hidden_size) {
    DEBUG_LOG("Position embedding shape is not expected.");
    return false;
  }

  // Get input "input_ids" from node.
  if (!CheckInput(input_ids, logger)) {
    DEBUG_LOG("Input id is not valid. ");
    return false;
  }

  NodeArg* gamma = layer_norm_node.MutableInputDefs()[1];
  NodeArg* beta = layer_norm_node.MutableInputDefs()[2];
  if (gamma->Shape() == nullptr || gamma->Shape()->dim()[0].dim_value() != hidden_size) {
    DEBUG_LOG("Gamma should be of shape (hidden_size). ");
    return false;
  }

  if (beta->Shape() == nullptr || beta->Shape()->dim()[0].dim_value() != hidden_size) {
    DEBUG_LOG("Beta should be of shape (hidden_size). ");
    return false;
  }

  CreateEmbedLayernormNode(graph, input_ids, nullptr, word_embedding, position_embedding, nullptr,
                           layer_norm_node);

  if (!nodes_to_remove.empty()) {
    graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *graph.GetNode(nodes_to_remove[0]));
  }

  nodes_to_remove.clear();

  nodes_to_remove.push_back(word_gather_node.Index());
  nodes_to_remove.push_back(add_node.Index());

  nodes_to_remove.push_back(layer_norm_node.Index());

  for (const NodeIndex index : nodes_to_remove) {
    Node* node = graph.GetNode(index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  return true;
}
/**
Embed Layer Normalization will fuse embeddings into one node :
The embeddings before conversion:
  (input_ids) -------->  Gather ---------+       (segment_ids)
    |                                    |           |
    |                                    v           v
    +--> Shape --> Expand -> Gather---->Add        Gather
    |                ^                    \         /
    |                |                     \       /
    +---(optional graph)                      Add
                                               |
                                       LayerNormalization
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
    // Find Attention after LayerNormalization
    const Node* p_attention = graph_utils::FirstChildByType(layer_norm_node, "Attention");
    if (p_attention == nullptr) {
      // Support model with multiple EmbedLayerNormalization.
      continue;
    }
    Node& attention_node = *graph.GetNode(p_attention->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(attention_node, "Attention", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(attention_node, GetCompatibleExecutionProviders())) {
      continue;
    }
    std::vector<const Node::EdgeEnd*> edges;

    // Find Add --> LayerNormalization
    if (!graph_utils::FindPath(layer_norm_node, true, {{0, 0, "Add", {7, 13}, kOnnxDomain}}, edges, logger)) {
      continue;
    }
    Node& layer_norm_add_node = *graph.GetNode(edges[0]->GetNode().Index());

    if (IsNeighborNodeExpectedTypes(layer_norm_add_node.InputEdgesBegin(), layer_norm_add_node.InputNodesEnd(), {"Gather", "Gather"})) {
      //DistilBert
      if (FuseSubGraphDistilBert(graph, layer_norm_add_node, layer_norm_node, logger)) {
        modified = true;
      }
    } else {
      if (FuseSubGraph(graph, layer_norm_add_node, layer_norm_node, modified, logger)) {
        modified = true;
      }
    }
  }
  return Status::OK();
}
}  // namespace onnxruntime
