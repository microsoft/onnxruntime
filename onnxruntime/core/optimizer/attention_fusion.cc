// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/attention_fusion.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/attention_fusion_helper.h"
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
    result.reserve(element_count);
    if (is_matmul) {
      MergeMatMulWeights<float>(q_weight, k_weight, v_weight, result, hidden_size);
    } else {
      MergeWeights<float>(q_weight, k_weight, v_weight, result, hidden_size);
    }
    initializer.set_raw_data(result.data(), element_count * sizeof(float));
  } else {  // data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
    const MLFloat16* q_weight = q_initializer.data<MLFloat16>();
    const MLFloat16* k_weight = k_initializer.data<MLFloat16>();
    const MLFloat16* v_weight = v_initializer.data<MLFloat16>();
    std::vector<MLFloat16> result;
    result.reserve(element_count);
    if (is_matmul) {
      MergeMatMulWeights<MLFloat16>(q_weight, k_weight, v_weight, result, hidden_size);
    } else {
      MergeWeights<MLFloat16>(q_weight, k_weight, v_weight, result, hidden_size);
    }
    initializer.set_raw_data(result.data(), element_count * sizeof(MLFloat16));
  }

  return graph_utils::AddInitializer(graph, initializer);
}

static NodeArg& AddMaskReduceSum(Graph& graph, NodeArg* reduce_sum_input, TypeProto& output_type, ProviderType provider_type) {
  NodeArg& reduce_sum_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MaskIndex_Int32"), &output_type);

  const std::vector<NodeArg*> input_defs{reduce_sum_input};
  const std::vector<NodeArg*> output_defs{&reduce_sum_output};
  Node& node = graph.AddNode(
      graph.GenerateNodeName("MaskIndex"),
      "ReduceSum",
      "Count number of words",
      input_defs,
      output_defs,
      {},
      kOnnxDomain);

  // Add attribute: "axes" = [1]
  ONNX_NAMESPACE::AttributeProto axes;
  axes.set_name("axes");
  axes.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  axes.add_ints(1);
  node.AddAttribute("axes", axes);

  // Add attribute: "keepdims" = 0
  ONNX_NAMESPACE::AttributeProto keepdims;
  keepdims.set_name("keepdims");
  keepdims.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  keepdims.set_i(static_cast<int64_t>(0));
  node.AddAttribute("keepdims", keepdims);

  node.SetExecutionProviderType(provider_type);

  return reduce_sum_output;
}

static NodeArg* ProcessMask(Graph& graph, NodeArg* mask_input, ProviderType provider_type, const logging::Logger& logger) {
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

  NodeArg* reduce_sum_input = mask_input;
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
    data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    NodeArg& cast_int32 = AttentionFusionHelper::CastMaskToInt32(graph, mask_input, provider_type);
    reduce_sum_input = &cast_int32;
  }

  // Construct shape based on mask input shape. Note that batch_size could be symbolic.
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  auto dim = output_type.mutable_tensor_type()->mutable_shape()->add_dim();
  *dim = mask_shape->dim(0);

  NodeArg& output = AddMaskReduceSum(graph, reduce_sum_input, output_type, provider_type);
  return &output;
}

static NodeArg* GetOrCreateMaskIndex(
    Graph& graph,
    NodeArg* mask_input,
    std::map<std::string, NodeArg*>& mask_index_map,
    ProviderType provider_type,
    const logging::Logger& logger) {
  // Lookup in map, and return the mask index if created.
  auto search = mask_index_map.find(mask_input->Name());
  if (search != mask_index_map.end()) {
    return search->second;
  }

  NodeArg* output = ProcessMask(graph, mask_input, provider_type, logger);
  if (nullptr == output) {
    return nullptr;
  }

  // Add it to map for lookup later.
  mask_index_map.insert(std::pair<std::string, NodeArg*>(mask_input->Name(), output));
  return output;
}

Status AttentionFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // A map from mask input arg name to mask index output.
  std::map<std::string, NodeArg*> mask_index_map;

  // A map from mask input arg name to the one casted to int32
  std::map<std::string, NodeArg*> mask_int32_map;

  int fused_count = 0;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (node.GetOutputEdgesCount() == 4 &&
        graph_utils::IsSupportedOptypeVersionAndDomain(node, "LayerNormalization", {1}, kOnnxDomain) &&
        graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      // Get hidden size from layer norm bias tensor shape.
      const NodeArg& layer_norm_bias = *(node.InputDefs()[2]);
      if (!optimizer_utils::IsShapeKnownOnAllDims(layer_norm_bias, 1)) {
        DEBUG_LOG("shape of layer norm bias tensor not expected");
        continue;
      }
      int64_t hidden_size = layer_norm_bias.Shape()->dim(0).dim_value();

      // Check that LayerNormalization has 4 children: 1 Add, 3 MatMul
      const Node* add_node = nullptr;
      int add_count = 0;
      int matmul_count = 0;
      int shape_count = 0;
      int reshape_count = 0;
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

      if (add_count == 1 && matmul_count == 3) { // BERT
        if (AttentionFusion::FuseSubGraph(node, *add_node, graph, hidden_size, mask_index_map, logger)) {
          fused_count++;
          modified = true;
        }
      } else if (reshape_count == 1 && shape_count == 3) {  // GPT
        if (AttentionFusionHelper::FuseGptAttention(node, graph, hidden_size, mask_int32_map, logger)) {
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
bool AttentionFusion::FuseSubGraph(Node& layer_norm, const Node& add_after_layer_norm, Graph& graph, int64_t hidden_size, std::map<std::string, NodeArg*>& mask_index_map, const logging::Logger& logger) {
  std::vector<graph_utils::EdgeEndToMatch> parent_path{
      {0, 0, "Add", {7}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9}, kOnnxDomain},
      {0, 0, "Reshape", {5}, kOnnxDomain},
      {0, 0, "Transpose", {1}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9}, kOnnxDomain},
      {0, 1, "Transpose", {1}, kOnnxDomain},
      {0, 0, "Reshape", {5}, kOnnxDomain},
      {0, 0, "Add", {7}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9}, kOnnxDomain},
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

  int64_t num_heads = 0;  // will be updated in CheckNodesInPathV
  int64_t head_size = 0;  // will be updated in CheckNodesInPathV
  if (!AttentionFusionHelper::CheckNodesInPathV(graph, reshape, transpose, qkv_matmul, v_transpose, v_reshape, num_heads, head_size, hidden_size, logger)) {
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

  // Find mask nodes: Unsqueeze -> Unsqueeze -> (Cast) -> Sub -> Mul -> Add -> Softmax --> [MatMul]
  // The "Cast" node in parentheses is optional.
  AttentionFusionHelper::AttentionMaskNodes mask_nodes;
  if (!AttentionFusionHelper::MatchInputMaskSubgraph(graph, qkv_matmul, mask_nodes, logger, false)) {
    DEBUG_LOG("Failed in match input mask subgraph");
    return false;
  }

  // path to q
  std::vector<graph_utils::EdgeEndToMatch> q_path{
      {0, 0, "Div", {7}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9}, kOnnxDomain},
      {0, 0, "Transpose", {1}, kOnnxDomain},
      {0, 0, "Reshape", {5}, kOnnxDomain},
      {0, 0, "Add", {7}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9}, kOnnxDomain},
      {0, 0, "LayerNormalization", {1}, kOnnxDomain}};

  if (!graph_utils::FindPath(*(mask_nodes.add), true, q_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for q");
    return false;
  }

  const Node& qk_div = edges[0]->GetNode();
  const Node& qk_matmul = edges[1]->GetNode();
  const Node& q_transpose = edges[2]->GetNode();
  const Node& q_reshape = edges[3]->GetNode();
  const Node& q_add = edges[4]->GetNode();
  const Node& q_matmul = edges[5]->GetNode();
  const Node& q_root = edges[6]->GetNode();
  if (q_root.Index() != layer_norm.Index()) {
    DEBUG_LOG("q root should be layer normalization");
    return false;
  }

  if (!AttentionFusionHelper::CheckNodesInPathQ(graph, qk_div, q_reshape, q_transpose, num_heads, head_size, logger)) {
    DEBUG_LOG("CheckNodesInPathQ returns false");
    return false;
  }

  if (!(ValidateAddBiasInitializer(graph, q_add, hidden_size) &&
        ValidateMatMulInitializer(graph, q_matmul, hidden_size))) {
    DEBUG_LOG("q_matmul and q_add shape not matched");
    return false;
  }

  // path to k
  std::vector<graph_utils::EdgeEndToMatch> k_path{
      {0, 1, "Transpose", {1}, kOnnxDomain},
      {0, 0, "Reshape", {5}, kOnnxDomain},
      {0, 0, "Add", {7}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9}, kOnnxDomain},
      {0, 0, "LayerNormalization", {1}, kOnnxDomain}};

  if (!graph_utils::FindPath(qk_matmul, true, k_path, edges, logger)) {
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

  // Load q, k and v weights
  const ONNX_NAMESPACE::TensorProto* q_weight_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* k_weight_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* v_weight_tensor = nullptr;
  if (!LoadQkvWeights(graph, q_matmul, k_matmul, v_matmul, q_weight_tensor, k_weight_tensor, v_weight_tensor)) {
    DEBUG_LOG("Failed to load Q, K and V weights, or data type is not float or float16.");
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* q_bias_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* k_bias_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* v_bias_tensor = nullptr;
  if (!LoadQkvWeights(graph, q_add, k_add, v_add, q_bias_tensor, k_bias_tensor, v_bias_tensor)) {
    DEBUG_LOG("Failed to load Q, K and V bias tensors, or data type is not float or float16.");
    return false;
  }

  // Now everything is ready, we will start fusing subgraph.
  NodeArg* mask_input = graph.GetNode(mask_nodes.unsqueeze_1->Index())->MutableInputDefs()[0];
  NodeArg* mask_index = GetOrCreateMaskIndex(graph, mask_input, mask_index_map, layer_norm.GetExecutionProviderType(), logger);
  if (nullptr == mask_index) {
    DEBUG_LOG("Failed to create mask index");
    return false;
  }

  // Merge Q, K and V weights
  NodeArg& qkv_weights = MergeQkvWeights(graph, hidden_size, q_weight_tensor, k_weight_tensor, v_weight_tensor, true);
  NodeArg& qkv_bias = MergeQkvWeights(graph, hidden_size, q_bias_tensor, k_bias_tensor, v_bias_tensor, false);

  // Create Attention Node.
  const std::vector<NodeArg*> input_defs{layer_norm.MutableOutputDefs()[0], &qkv_weights, &qkv_bias, mask_index};
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
  std::vector<NodeIndex> nodes_to_remove{
      reshape.Index(),
      transpose.Index(),
      qkv_matmul.Index(),
      v_transpose.Index(),
      v_reshape.Index(),
      v_add.Index(),
      v_matmul.Index(),
      qk_div.Index(),
      qk_matmul.Index(),
      q_transpose.Index(),
      q_reshape.Index(),
      q_add.Index(),
      q_matmul.Index(),
      k_transpose.Index(),
      k_reshape.Index(),
      k_add.Index(),
      k_matmul.Index()};

  AttentionFusionHelper::SetMaskNodesToRemove(graph, mask_nodes, nodes_to_remove);

  for (const auto& node_index : nodes_to_remove) {
    Node* node = graph.GetNode(node_index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  DEBUG_LOG("Fused an attention node.");

  return true;
}

}  // namespace onnxruntime
