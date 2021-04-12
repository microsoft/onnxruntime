// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/attention_fusion.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/attention_fusion_helper.h"
#include "core/graph/graph_utils.h"
#include <cmath>

namespace onnxruntime {

static bool ValidateMatMulInitializer(const Graph& graph, const Node& matmul, int64_t hidden_size) {
  const NodeArg& input_b = *(matmul.InputDefs()[1]);
  if (!graph_utils::IsInitializer(graph, input_b.Name(), true)) {
    return false;
  }

  return optimizer_utils::ValidateShape(input_b, {hidden_size, hidden_size});
}

static bool ValidateAddBiasInitializer(const Graph& graph, const Node& add, int64_t hidden_size) {
  const NodeArg& input_b = *(add.InputDefs()[1]);
  if (!graph_utils::IsInitializer(graph, input_b.Name(), true)) {
    return false;
  }

  return optimizer_utils::ValidateShape(input_b, {hidden_size});
}

// Merge 1-D weights (q, k and v) by concatenating them one by one.
template <typename T>
void MergeWeights(const T* q, const T* k, const T* v, std::vector<T>& result, int64_t element_count) {
  for (int64_t i = 0; i < element_count; i++) {
    result.push_back(*q);
    q++;
  }

  for (int64_t i = 0; i < element_count; i++) {
    result.push_back(*k);
    k++;
  }

  for (int64_t i = 0; i < element_count; i++) {
    result.push_back(*v);
    v++;
  }
}

// Merge 2-D weights (q, k and v) by concatenating them row by row.
template <typename T>
void MergeMatMulWeights(const T* q_weight, const T* k_weight, const T* v_weight, std::vector<T>& result, int64_t hidden_size) {
  const T* q = q_weight;
  const T* k = k_weight;
  const T* v = v_weight;
  for (int64_t i = 0; i < hidden_size; i++, q += hidden_size, k += hidden_size, v += hidden_size) {
    MergeWeights(q, k, v, result, hidden_size);
  }
}

// Load q, k and v weights, and validate their data types.
static bool LoadQkvWeights(
    Graph& graph,
    const Node& q, const Node& k, const Node& v,
    const ONNX_NAMESPACE::TensorProto*& q_tensor,
    const ONNX_NAMESPACE::TensorProto*& k_tensor,
    const ONNX_NAMESPACE::TensorProto*& v_tensor) {
  if (!graph.GetInitializedTensor(q.InputDefs()[1]->Name(), q_tensor)) {
    return false;
  }

  // Attention Op requires float or float16 weights.
  const auto data_type = q_tensor->data_type();
  if (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return false;
  }

  if (!graph.GetInitializedTensor(k.InputDefs()[1]->Name(), k_tensor) ||
      data_type != k_tensor->data_type()) {
    return false;
  }

  if (!graph.GetInitializedTensor(v.InputDefs()[1]->Name(), v_tensor) ||
      data_type != v_tensor->data_type()) {
    return false;
  }

  return true;
}

// Merge the weights of Q, K and V inputs for MatMul or Add (bias) into one input.
static NodeArg& MergeQkvWeights(Graph& graph, int64_t hidden_size,
                                const ONNX_NAMESPACE::TensorProto* q_tensor,
                                const ONNX_NAMESPACE::TensorProto* k_tensor,
                                const ONNX_NAMESPACE::TensorProto* v_tensor,
                                bool is_matmul) {
  assert(nullptr != q_tensor);
  assert(nullptr != k_tensor);
  assert(nullptr != v_tensor);
  Initializer q_initializer(*q_tensor, graph.ModelPath());
  Initializer k_initializer(*k_tensor, graph.ModelPath());
  Initializer v_initializer(*v_tensor, graph.ModelPath());
  auto data_type = q_tensor->data_type();

  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name(graph.GenerateNodeArgName(is_matmul ? "qkv_weights" : "qkv_bias"));
  // Shape of weights for MatMul is (hidden_size, 3 * hidden_size)
  // Shape of weights for Add bias is (3 * hidden_size)
  if (is_matmul) {
    initializer.add_dims(hidden_size);
  }
  initializer.add_dims(3 * hidden_size);
  initializer.set_data_type(data_type);
  const int64_t element_count = 3 * hidden_size * (is_matmul ? hidden_size : 1);

  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const float* q_weight = q_initializer.data<float>();
    const float* k_weight = k_initializer.data<float>();
    const float* v_weight = v_initializer.data<float>();
    std::vector<float> result;
    result.reserve(gsl::narrow<size_t>(element_count));
    if (is_matmul) {
      MergeMatMulWeights<float>(q_weight, k_weight, v_weight, result, hidden_size);
    } else {
      MergeWeights<float>(q_weight, k_weight, v_weight, result, hidden_size);
    }
    initializer.set_raw_data(result.data(), gsl::narrow<size_t>(element_count) * sizeof(float));
  } else {  // data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
    const MLFloat16* q_weight = q_initializer.data<MLFloat16>();
    const MLFloat16* k_weight = k_initializer.data<MLFloat16>();
    const MLFloat16* v_weight = v_initializer.data<MLFloat16>();
    std::vector<MLFloat16> result;
    result.reserve(gsl::narrow<size_t>(element_count));
    if (is_matmul) {
      MergeMatMulWeights<MLFloat16>(q_weight, k_weight, v_weight, result, hidden_size);
    } else {
      MergeWeights<MLFloat16>(q_weight, k_weight, v_weight, result, hidden_size);
    }
    initializer.set_raw_data(result.data(), gsl::narrow<size_t>(element_count) * sizeof(MLFloat16));
  }

  return graph_utils::AddInitializer(graph, initializer);
}

static NodeArg* ConvertMaskToInt32(Graph& graph, NodeArg* mask_input, ProviderType provider_type, const logging::Logger& logger) {
  // Validate mask input shape (batch_size, sequence_length) and data type.
  // Note that batch_size and sequence_length could be symbolic.
  const TensorShapeProto* mask_shape = mask_input->Shape();
  if (mask_shape == nullptr || mask_shape->dim_size() != 2 || mask_input->Type() == nullptr) {
    DEBUG_LOG("Mask shape is unknown or not 2D, or data type unknown");
    return nullptr;
  }

  auto data_type = mask_input->TypeAsProto()->tensor_type().elem_type();
  if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32 &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    DEBUG_LOG("Mask data type is not int32 or int64 or float32");
    return nullptr;
  }

  NodeArg* mask_int32 = mask_input;
  if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    NodeArg& cast_int32 = AttentionFusionHelper::CastMaskToInt32(graph, mask_input, provider_type);
    mask_int32 = &cast_int32;
  }

  return mask_int32;
}

static NodeArg* ConvertMaskToInt32(
    Graph& graph,
    NodeArg* mask_input,
    std::map<std::string, NodeArg*>& mask_int32_map,
    ProviderType provider_type,
    const logging::Logger& logger) {
  // Lookup in map, and return the converted mask.
  auto search = mask_int32_map.find(mask_input->Name());
  if (search != mask_int32_map.end()) {
    return search->second;
  }

  NodeArg* output = ConvertMaskToInt32(graph, mask_input, provider_type, logger);
  if (nullptr == output) {
    return nullptr;
  }

  // Add it to map for lookup later.
  mask_int32_map.insert(std::pair<std::string, NodeArg*>(mask_input->Name(), output));
  return output;
}

Status AttentionFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // A map from mask input arg name to the one casted to int32
  std::map<std::string, NodeArg*> mask_int32_map;

  int fused_count = 0;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if ((node.GetOutputEdgesCount() >= 2 && node.GetOutputEdgesCount() <= 6) &&  // Add node.GetOutputEdgesCount() == 5/6 for distilbert
        graph_utils::IsSupportedOptypeVersionAndDomain(node, "LayerNormalization", {1}, kOnnxDomain) &&
        graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      // Get hidden size from layer norm bias tensor shape.
      const NodeArg& layer_norm_bias = *(node.InputDefs()[2]);
      if (!optimizer_utils::IsShapeKnownOnAllDims(layer_norm_bias, 1)) {
        DEBUG_LOG("shape of layer norm bias tensor not expected");
        continue;
      }
      int64_t hidden_size = layer_norm_bias.Shape()->dim(0).dim_value();

      const Node* add_node = nullptr;
      unsigned int add_count = 0;
      unsigned int matmul_count = 0;
      unsigned int shape_count = 0;
      unsigned int reshape_count = 0;
      for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
        if ((*it).OpType().compare("Add") == 0) {
          add_count++;
          add_node = &(*it);
        } else if ((*it).OpType().compare("MatMul") == 0) {
          matmul_count++;
        } else if ((*it).OpType().compare("Shape") == 0) {
          shape_count++;
        } else if ((*it).OpType().compare("Reshape") == 0) {
          reshape_count++;
        }
      }

      if (add_count == 1 && matmul_count == 3 && shape_count == node.GetOutputEdgesCount() - 4) {  // BERT or DistilBert
        if (AttentionFusion::FuseSubGraph(node, *add_node, graph, hidden_size, mask_int32_map, logger)) {
          fused_count++;
          modified = true;
        }
      } else if (reshape_count == 1 && (shape_count == 1 || shape_count == 3) && (reshape_count + shape_count) == node.GetOutputEdgesCount()) {  // GPT
        if (AttentionFusionHelper::FuseGptAttention(node, graph, hidden_size, mask_int32_map, shape_count == 1, logger)) {
          fused_count++;
          modified = true;
        }
      }
    }
  }

  if (fused_count > 0) {
    LOGS(logger, INFO) << "Total fused Attention node count: " << fused_count;
  }

  return Status::OK();
}

static bool FuseSubGraphQKImpl(Node& layer_norm,
                               Graph& graph,
                               std::vector<std::reference_wrapper<const Node>>& parent_path_nodes,
                               NodeArg* mask_input,
                               std::map<std::string, NodeArg*>& mask_int32_map,
                               std::vector<const Node::EdgeEnd*>& edges,
                               std::vector<NodeIndex>& nodes_to_remove,
                               int64_t hidden_size,
                               int64_t num_heads,
                               int64_t head_size,
                               const logging::Logger& logger) {
  std::vector<std::reference_wrapper<const Node>> pivot_nodes;
  if (edges.size() == 2) {
    const Node& qk_div = (edges[0]->GetNode().OpType() == "Div") ? edges[0]->GetNode() : edges[1]->GetNode();
    const Node& qk_matmul = (edges[1]->GetNode().OpType() == "MatMul") ? edges[1]->GetNode() : edges[0]->GetNode();
    pivot_nodes.push_back(qk_matmul);
    pivot_nodes.push_back(qk_div);
  } else {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> q_path{
      {0, 0, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Add", {7, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "LayerNormalization", {1}, kOnnxDomain}};
  if (!graph_utils::FindPath(edges[edges.size() - 1]->GetNode(), true, q_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for q");
    return false;
  }

  const Node& q_transpose = edges[0]->GetNode();
  const Node& q_reshape = edges[1]->GetNode();
  const Node& q_add = edges[2]->GetNode();
  const Node& q_matmul = edges[3]->GetNode();
  const Node& q_root = edges[4]->GetNode();
  if (q_root.Index() != layer_norm.Index()) {
    DEBUG_LOG("q root should be layer normalization");
    return false;
  }

  if (!AttentionFusionHelper::CheckNodesInPathQ(graph, pivot_nodes[1].get(), q_reshape, q_transpose, num_heads, head_size, logger)) {
    DEBUG_LOG("CheckNodesInPathQ returns false");
    return false;
  }

  if (!(ValidateAddBiasInitializer(graph, q_add, hidden_size) &&
        ValidateMatMulInitializer(graph, q_matmul, hidden_size))) {
    DEBUG_LOG("q_matmul and q_add shape not matched");
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> k_path{
      {0, 1, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Add", {7, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "LayerNormalization", {1}, kOnnxDomain}};

  if (!graph_utils::FindPath(pivot_nodes[0].get(), true, k_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for k");
    return false;
  }

  const Node& k_transpose = edges[0]->GetNode();
  const Node& k_reshape = edges[1]->GetNode();
  const Node& k_add = edges[2]->GetNode();
  const Node& k_matmul = edges[3]->GetNode();
  const Node& k_root = edges[4]->GetNode();
  if (k_root.Index() != layer_norm.Index()) {
    DEBUG_LOG("k root is not layer norm");
    return false;
  }
  if (!AttentionFusionHelper::CheckNodesInPathK(graph, k_reshape, k_transpose, num_heads, head_size, logger)) {
    DEBUG_LOG("CheckNodesInPathK returns false");
    return false;
  }

  if (!(ValidateAddBiasInitializer(graph, k_add, hidden_size) &&
        ValidateMatMulInitializer(graph, k_matmul, hidden_size))) {
    DEBUG_LOG("k_matmul and k_add shape not matched");
    return false;
  }

  const Node& v_matmul = parent_path_nodes[6];
  // Load q, k and v weights
  const ONNX_NAMESPACE::TensorProto* q_weight_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* k_weight_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* v_weight_tensor = nullptr;
  if (!LoadQkvWeights(graph, q_matmul, k_matmul, v_matmul, q_weight_tensor, k_weight_tensor, v_weight_tensor)) {
    DEBUG_LOG("Failed to load Q, K and V weights, or data type is not float or float16.");
    return false;
  }

  const Node& v_add = parent_path_nodes[5];
  const ONNX_NAMESPACE::TensorProto* q_bias_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* k_bias_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* v_bias_tensor = nullptr;
  if (!LoadQkvWeights(graph, q_add, k_add, v_add, q_bias_tensor, k_bias_tensor, v_bias_tensor)) {
    DEBUG_LOG("Failed to load Q, K and V bias tensors, or data type is not float or float16.");
    return false;
  }

  // Now everything is ready, we will start fusing subgraph.
  NodeArg* mask_int32 = ConvertMaskToInt32(graph, mask_input, mask_int32_map, layer_norm.GetExecutionProviderType(), logger);
  if (nullptr == mask_int32) {
    DEBUG_LOG("Failed to convert mask to int32");
    return false;
  }

  // Merge Q, K and V weights
  NodeArg& qkv_weights = MergeQkvWeights(graph, hidden_size, q_weight_tensor, k_weight_tensor, v_weight_tensor, true);
  NodeArg& qkv_bias = MergeQkvWeights(graph, hidden_size, q_bias_tensor, k_bias_tensor, v_bias_tensor, false);
  // Create Attention Node.
  const Node& reshape = parent_path_nodes[0];
  const std::vector<NodeArg*> input_defs{layer_norm.MutableOutputDefs()[0], &qkv_weights, &qkv_bias, mask_int32};
  const std::vector<NodeArg*> output_defs{graph.GetNode(reshape.Index())->MutableOutputDefs()[0]};
  Node& attention_node = graph.AddNode(
      graph.GenerateNodeName("Attention"),
      "Attention",
      "Fused Attention subgraphs ",
      input_defs,
      output_defs,
      nullptr,
      kMSDomain);
  attention_node.AddAttribute("num_heads", num_heads);

  // Assign provider to this new node.
  attention_node.SetExecutionProviderType(layer_norm.GetExecutionProviderType());

  // Remove nodes that are not used anymore.
  parent_path_nodes.insert(parent_path_nodes.end(), pivot_nodes.begin(), pivot_nodes.end());

  std::transform(parent_path_nodes.begin(),
                 parent_path_nodes.end(),
                 std::back_inserter(nodes_to_remove),
                 [](std::reference_wrapper<const Node> node_ref_wrapper) -> NodeIndex {
                   return node_ref_wrapper.get().Index();
                 });

  std::vector<NodeIndex> nodes_to_remove_temp{
      q_transpose.Index(),
      q_reshape.Index(),
      q_add.Index(),
      q_matmul.Index(),
      k_transpose.Index(),
      k_reshape.Index(),
      k_add.Index(),
      k_matmul.Index()};

  nodes_to_remove.insert(nodes_to_remove.end(), nodes_to_remove_temp.begin(), nodes_to_remove_temp.end());

  return true;
}

static bool FuseSubGraphQK(Node& layer_norm,
                           Graph& graph,
                           AttentionFusionHelper::AttentionMaskNodes& mask_nodes,
                           NodeArg* mask_input,
                           std::vector<std::reference_wrapper<const Node>>& parent_path_nodes,
                           int64_t hidden_size,
                           int64_t num_heads,
                           int64_t head_size,
                           std::map<std::string, NodeArg*>& mask_int32_map,
                           const logging::Logger& logger) {
  // path to q
  std::vector<graph_utils::EdgeEndToMatch> q_varience_path{
      {0, 0, "Div", {7, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(*(mask_nodes.add), true, q_varience_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for q");
    return false;
  }

  std::vector<NodeIndex> nodes_to_remove;
  if (!FuseSubGraphQKImpl(layer_norm, graph, parent_path_nodes, mask_input, mask_int32_map, edges, nodes_to_remove, hidden_size,
                          num_heads, head_size, logger)) {
    return false;
  }

  AttentionFusionHelper::SetMaskNodesToRemove(graph, mask_nodes, nodes_to_remove);

  for (const auto& node_index : nodes_to_remove) {
    Node* node = graph.GetNode(node_index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  DEBUG_LOG("Fused an attention node.");

  return true;
}

/** DistilBert's attention is a bit different here
@remark add_after_layer_norm is the Add node in the bottom of sub-graph.
 Abbreviatios: B is batch_size, S is sequence_length, W is hidden_size
               N is number of attention heads, H is head size, and W=N*H
               B and S could be symbolic.
    Graph before Fusion (q_, k_, v_, qk_, qkv_ and mask_ prefix is added before Operator type):
                  [Input](BxSxW)
                        |
                LayerNormalization ---------------------------------------------
            /       |        |     \     [Weights](WxW)                         |
           /        |        |      \    /                                    Shape
          |   q_MatMul    k_MatMul  v_MatMul  [Bias](W)                     /       \
          |         |        |        |   /                                /         \
          |     q_Add     k_Add     v_Add     [Shape=0,-1,N,H]    Gather(indices:0)   Gather(indices:1)
          |         |        |        |      /                          |               |
          | q_Reshape   k_Reshape   v_Reshape                           |               |
          |         |        |        |                             Unsqueeze        Unsqueeze
          |q_Transpose  k_Transpose v_Transpose                         |   \          /
          |  (0,2,1,3)  (0,2,3,1)    (perm=0,2,1,3)                     |    \        /
          |         |       |         |                                 |     \      /
          |        q_Div   /                                            |      Concat [_, 1, 1, _]
          |           |  /            |                                 |         |
          |        qk_MatMul          |                                 |         |         --------- AttentionMask
          |           |    \          |                                 |         |        /
          |           |      \        |                                 |         |       /
          |           |     Shape     |                                 |         |     Equal (B = 0)
          |           |       |       |                                 |         |     /
          |           |    Expand-----|-----------------------------------------Reshape
          |            \   /          |                                 |
          |         Where             /                                 |
          |             |           /                                   |
          |          Softmax       /                                    |
          |             \         /                                     |
          |              \       /                                      |
          |            qkv_MatMul                                       |
          |                   |                                         |
          |                Transpose (perm=0,2,1,3)                     |
          |                   |                                         |
          |                Reshape-----------------------------------Concat [Shape=_,-1,W]
          |                   |
          |                 MatMul----[Weights](WxW)
          |                   |
          |                  Add----[Bias](W)
          +-------------------|---+
                              |   |
                               Add

A change compared with first version attention fusion for distilbert:
There were two Shape nodes after LayerNormalization, gets fused into one before ORT 1.5.0 release

However, the first version of attention fusion for distilbert is still supported for now.
*/
static bool FuseSubGraphQKDistilBert(Node& layer_norm,
                                     Graph& graph,
                                     AttentionFusionHelper::AttentionMaskNodesDistilBert& mask_nodes,
                                     NodeArg* mask_input,
                                     std::vector<std::reference_wrapper<const Node>>& parent_path_nodes,
                                     int64_t hidden_size,
                                     int64_t num_heads,
                                     int64_t head_size,
                                     std::map<std::string, NodeArg*>& mask_int32_map,
                                     const logging::Logger& logger) {
  // path to q
  std::vector<graph_utils::EdgeEndToMatch> q_varience_path{
      {0, 2, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "Div", {7, 13}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(*(mask_nodes.where), true, q_varience_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for q");
    return false;
  }

  std::vector<NodeIndex> nodes_to_remove;
  if (!FuseSubGraphQKImpl(layer_norm, graph, parent_path_nodes, mask_input, mask_int32_map, edges, nodes_to_remove, hidden_size,
                          num_heads, head_size, logger)) {
    return false;
  }

  const Node& reshape_1 = parent_path_nodes[0];
  const Node& reshape_2 = *(mask_nodes.reshape);

  const Node* p_concat_1 = graph_utils::GetInputNode(reshape_1, 1);
  const Node* p_concat_2 = graph_utils::GetInputNode(reshape_2, 1);
  if (p_concat_1 != nullptr && p_concat_2 != nullptr) {
    graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *p_concat_1);
    graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *p_concat_2);
  } else {
    return false;
  }

  AttentionFusionHelper::SetMaskNodesToRemove(graph, mask_nodes, nodes_to_remove);

  for (const auto& node_index : nodes_to_remove) {
    Node* node = graph.GetNode(node_index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  DEBUG_LOG("Fused an attention node.");

  return true;
}

/** Fuse Attention SubGraph.
@remark add_after_layer_norm is the Add node in the bottom of sub-graph.
 Abbreviatios: B is batch_size, S is sequence_length, W is hidden_size
               N is number of attention heads, H is head size, and W=N*H
               B and S could be symbolic.
    Graph before Fusion (q_, k_, v_, qk_, qkv_ and mask_ prefix is added before Operator type):
                  [Input](BxSxW)
                        |
                LayerNormalization
            /       |        |     \     [Weights](WxW)
           /        |        |      \    /
          |   q_MatMul    k_MatMul  v_MatMul  [Bias](W)
          |         |        |        |   /
          |     q_Add     k_Add     v_Add     [Shape=0,0,N,H]
          |         |        |        |      /
          | q_Reshape   k_Reshape   v_Reshape                [Mask] (BxS)
          |         |        |        |                          |
          |q_Transpose  k_Transpose v_Transpose            mask_Unsqueeze(axes=1)
          |  (0,2,1,3)  (0,2,3,1)    (perm=0,2,1,3)              |
          |         \       /         |                    mask_Unsqueeze(axes=2)
          |      qk_MatMul            |                          |
          |           |    [B=2]      |              ([A=1.0] mask_Cast(to=1))
          |           |   /           |                   \     /
          |        qk_Div             |                 mask_Sub   [B=-10000.0]
          |            \              |                        \   /
          |       mask_Add <-------- /---------------------mask_Mul
          |             |           /
          |          Softmax       /
          |             \         /
          |              \       /
          |            qkv_MatMul
          |                   |
          |                Transpose (perm=0,2,1,3)
          |                   |
          |                Reshape---[shape=0,0,W]
          |                   |
          |                 MatMul----[Weights](WxW)
          |                   |
          |                  Add----[Bias](W)
          +-------------------|---+
                              |   |
                               Add

After Fusion:
  LayerNormalization  [Weights](Wx3W)   Mask
      |        \      /   [Bias](3W)     |
      |         \    /   /               |
      |         Attention <------------ReduceSum
      \          |
       \        MatMul
        \        |
         \      Add
          +------|---+
                 |   |
                  Add
*/
bool AttentionFusion::FuseSubGraph(Node& layer_norm, const Node& add_after_layer_norm, Graph& graph, int64_t hidden_size, std::map<std::string, NodeArg*>& mask_int32_map, const logging::Logger& logger) {
  std::vector<graph_utils::EdgeEndToMatch> parent_path{
      {0, 0, "Add", {7, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 1, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Add", {7, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "LayerNormalization", {1}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(add_after_layer_norm, true, parent_path, edges, logger)) {
    DEBUG_LOG("Faild to find path v");
    return false;
  }

  const Node& add = edges[0]->GetNode();
  const Node& matmul = edges[1]->GetNode();
  const Node& reshape = edges[2]->GetNode();
  const Node& transpose = edges[3]->GetNode();
  const Node& qkv_matmul = edges[4]->GetNode();
  const Node& v_transpose = edges[5]->GetNode();
  const Node& v_reshape = edges[6]->GetNode();
  const Node& v_add = edges[7]->GetNode();
  const Node& v_matmul = edges[8]->GetNode();
  const Node& v_root = edges[9]->GetNode();
  if (v_root.Index() != layer_norm.Index()) {
    return false;
  }

  if (!optimizer_utils::CheckOutputEdges(graph, v_add, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, v_matmul, 1)) {
    DEBUG_LOG("Output edge count not expected for Add or MatMul in path v");
    return false;
  }

  int64_t num_heads = 0;          // will be updated in CheckNodesInPathV
  int64_t head_size = 0;          // will be updated in CheckNodesInPathV
  NodeIndex record_node_idx = 0;  // will be updated in CheckNodesInPathV if it's distilbert model
  if (!AttentionFusionHelper::CheckNodesInPathV(graph, reshape, transpose, qkv_matmul, v_transpose, v_reshape, num_heads, head_size, hidden_size, record_node_idx, logger)) {
    DEBUG_LOG("CheckNodesInPathV return false");
    return false;
  }

  // Validate the input shape of MatMul and Add according to hidden_size.
  if (!(ValidateAddBiasInitializer(graph, add, hidden_size) &&
        ValidateMatMulInitializer(graph, matmul, hidden_size) &&
        ValidateAddBiasInitializer(graph, v_add, hidden_size) &&
        ValidateMatMulInitializer(graph, v_matmul, hidden_size))) {
    DEBUG_LOG("Failed in match v_matmul and v_add input shape");
    return false;
  }

  //store parent path
  std::vector<std::reference_wrapper<const Node>> parent_path_nodes{reshape, transpose, qkv_matmul, v_transpose, v_reshape, v_add, v_matmul};

  // Find mask nodes: Unsqueeze -> Unsqueeze -> (Cast) -> Sub -> Mul -> Add -> Softmax --> [MatMul]
  // The "Cast" node in parentheses is optional.
  AttentionFusionHelper::AttentionMaskNodes mask_nodes;
  AttentionFusionHelper::AttentionMaskNodesDistilBert mask_nodes_distilbert;

  if (AttentionFusionHelper::MatchInputMaskSubgraph(graph, qkv_matmul, mask_nodes, logger, false)) {
    NodeArg* mask_input = graph.GetNode(mask_nodes.unsqueeze_1->Index())->MutableInputDefs()[0];
    return FuseSubGraphQK(layer_norm, graph, mask_nodes, mask_input, parent_path_nodes, hidden_size, num_heads, head_size, mask_int32_map, logger);
  } else if (AttentionFusionHelper::MatchInputMaskSubgraph(graph, layer_norm, qkv_matmul, mask_nodes_distilbert, record_node_idx, logger)) {
    NodeArg* mask_input = graph.GetNode(mask_nodes_distilbert.equal->Index())->MutableInputDefs()[0];
    return FuseSubGraphQKDistilBert(layer_norm, graph, mask_nodes_distilbert, mask_input, parent_path_nodes, hidden_size, num_heads, head_size, mask_int32_map, logger);
  } else {
    DEBUG_LOG("Failed in match input mask subgraph");
    return false;
  }

  return true;
}

}  // namespace onnxruntime
