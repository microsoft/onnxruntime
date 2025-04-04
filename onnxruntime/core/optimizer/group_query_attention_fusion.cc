// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/group_query_attention_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

template <typename T>
void MergeRows(const T* weight, std::vector<T>& result, int64_t N, int64_t blocks, int64_t block_size) {
  result.insert(result.end(), weight, weight + N * blocks * block_size);
}

// Merge Q, K, V tensors into a single vector in the order:
// [N rows of Q, N rows of K, N rows of V].
template <typename T>
void MergeMatMulWeights(const T* q_weight, const T* k_weight, const T* v_weight,
                        std::vector<T>& result, int64_t N, int64_t blocks, int64_t block_size) {
  MergeRows(q_weight, result, N, blocks, block_size);
  MergeRows(k_weight, result, N, blocks, block_size);
  MergeRows(v_weight, result, N, blocks, block_size);
}

static void AttachNodeAttribute(Node& node, const std::string& attribute_name, int64_t attribute_value, AttributeProto_AttributeType attribute_type) {
  NodeAttributes& node_attributes = node.GetMutableAttributes();
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name(attribute_name);
  attr.set_i(attribute_value);
  attr.set_type(attribute_type);
  node_attributes[attribute_name] = attr;
}

static NodeArg& MergeQkvWeights2(Graph& graph,
                                               const ONNX_NAMESPACE::TensorProto* q_tensor,
                                               const ONNX_NAMESPACE::TensorProto* k_tensor,
                                               const ONNX_NAMESPACE::TensorProto* v_tensor) {
  assert(nullptr != q_tensor);
  assert(nullptr != k_tensor);
  assert(nullptr != v_tensor);

  Initializer q_initializer(*q_tensor, graph.ModelPath());
  Initializer k_initializer(*k_tensor, graph.ModelPath());
  Initializer v_initializer(*v_tensor, graph.ModelPath());

  ONNX_NAMESPACE::TensorProto qkv_b_initializer;
  qkv_b_initializer.set_name(graph.GenerateNodeArgName("qkv_B"));
  qkv_b_initializer.add_dims(2304);
  qkv_b_initializer.add_dims(4096);
  qkv_b_initializer.set_data_type(q_tensor->data_type());

  const MLFloat16* q_data = q_initializer.data<MLFloat16>();
  const MLFloat16* k_data = k_initializer.data<MLFloat16>();
  const MLFloat16* v_data = v_initializer.data<MLFloat16>();

  size_t b_element_count = 2304 * 4096;
  std::vector<MLFloat16> merged_qkv_B;
  merged_qkv_B.reserve(b_element_count);

  optimizer_utils::MergeMatMulWeights(q_data, k_data, v_data, merged_qkv_B, 2304, 2048, 1024);

  utils::SetRawDataInTensorProto(qkv_b_initializer, merged_qkv_B.data(), b_element_count * sizeof(MLFloat16));

  NodeArg& qkv_b_arg = graph_utils::AddInitializer(graph, qkv_b_initializer);

  return qkv_b_arg;
}

/*
static std::array<NodeArg*, 3> MergeQkvWeights(Graph& graph,
                                               int64_t hidden_dim,
                                               int64_t blocks,
                                               int64_t block_size,
                                               const ONNX_NAMESPACE::TensorProto* q_tensor,
                                               const ONNX_NAMESPACE::TensorProto* k_tensor,
                                               const ONNX_NAMESPACE::TensorProto* v_tensor,
                                               const ONNX_NAMESPACE::TensorProto* q_scale_tensor,
                                               const ONNX_NAMESPACE::TensorProto* q_zero_point_tensor,
                                               const ONNX_NAMESPACE::TensorProto* k_scale_tensor,
                                               const ONNX_NAMESPACE::TensorProto* k_zero_point_tensor,
                                               const ONNX_NAMESPACE::TensorProto* v_scale_tensor,
                                               const ONNX_NAMESPACE::TensorProto* v_zero_point_tensor) {
  assert(nullptr != q_tensor);
  assert(nullptr != k_tensor);
  assert(nullptr != v_tensor);

  Initializer q_initializer(*q_tensor, graph.ModelPath());
  Initializer k_initializer(*k_tensor, graph.ModelPath());
  Initializer v_initializer(*v_tensor, graph.ModelPath());

  Initializer q_scale_initializer(*q_scale_tensor, graph.ModelPath());
  Initializer q_zp_initializer(*q_zero_point_tensor, graph.ModelPath());

  Initializer k_scale_initializer(*k_scale_tensor, graph.ModelPath());
  Initializer k_zp_initializer(*k_zero_point_tensor, graph.ModelPath());

  Initializer v_scale_initializer(*v_scale_tensor, graph.ModelPath());
  Initializer v_zp_initializer(*v_zero_point_tensor, graph.ModelPath());

  ONNX_NAMESPACE::TensorProto qkv_b_initializer;
  qkv_b_initializer.set_name(graph.GenerateNodeArgName("qkv_B"));
  qkv_b_initializer.add_dims(3 * hidden_dim);
  qkv_b_initializer.add_dims(blocks);
  qkv_b_initializer.add_dims(block_size);
  qkv_b_initializer.set_data_type(q_tensor->data_type());

  ONNX_NAMESPACE::TensorProto qkv_scale_initializer;
  qkv_scale_initializer.set_name(graph.GenerateNodeArgName("qkv_scale"));
  // Preserve the original tensor dimensionality.
  if (q_scale_tensor->dims().size() > 1) {
    qkv_scale_initializer.add_dims(3 * hidden_dim);
    qkv_scale_initializer.add_dims(blocks);
  } else {
    qkv_scale_initializer.add_dims(3 * hidden_dim * blocks);
  }

  qkv_scale_initializer.set_data_type(q_scale_tensor->data_type());

  ONNX_NAMESPACE::TensorProto qkv_zp_initializer;
  qkv_zp_initializer.set_name(graph.GenerateNodeArgName("qkv_zp"));

  // The number of zp elements is half of the scale elements size.
  qkv_zp_initializer.add_dims(3 * hidden_dim * blocks / 2);
  qkv_zp_initializer.set_data_type(q_zero_point_tensor->data_type());

  const uint8_t* q_data = q_initializer.data<uint8_t>();
  const uint8_t* k_data = k_initializer.data<uint8_t>();
  const uint8_t* v_data = v_initializer.data<uint8_t>();

  const MLFloat16* q_scale_data = q_scale_initializer.data<MLFloat16>();
  const uint8_t* q_zero_points_data = q_zp_initializer.data<uint8_t>();
  const MLFloat16* k_scale_data = k_scale_initializer.data<MLFloat16>();
  const uint8_t* k_zero_points_data = k_zp_initializer.data<uint8_t>();
  const MLFloat16* v_scale_data = v_scale_initializer.data<MLFloat16>();
  const uint8_t* v_zero_points_data = v_zp_initializer.data<uint8_t>();

  size_t b_element_count = 3 * hidden_dim * blocks * block_size;
  std::vector<uint8_t> merged_qkv_B;
  merged_qkv_B.reserve(b_element_count);

  size_t scale_elements_count = 3 * hidden_dim * blocks;
  std::vector<MLFloat16> merged_qkv_scale;
  merged_qkv_scale.reserve(scale_elements_count);

  size_t zp_elements_count = 3 * hidden_dim * blocks / 2;
  std::vector<uint8_t> merged_qkv_zp;
  merged_qkv_zp.reserve(zp_elements_count);

  MergeMatMulWeights(q_data, k_data, v_data, merged_qkv_B, hidden_dim, blocks, block_size);
  MergeMatMulWeights(q_scale_data, k_scale_data, v_scale_data, merged_qkv_scale, hidden_dim, blocks, 1);
  MergeMatMulWeights(q_zero_points_data, k_zero_points_data, v_zero_points_data, merged_qkv_zp, hidden_dim, blocks / 2, 1);

  utils::SetRawDataInTensorProto(qkv_b_initializer, merged_qkv_B.data(), b_element_count * sizeof(uint8_t));
  utils::SetRawDataInTensorProto(qkv_scale_initializer, merged_qkv_scale.data(), scale_elements_count * sizeof(MLFloat16));
  utils::SetRawDataInTensorProto(qkv_zp_initializer, merged_qkv_zp.data(), zp_elements_count * sizeof(uint8_t));

  NodeArg& qkv_b_arg = graph_utils::AddInitializer(graph, qkv_b_initializer);
  NodeArg& qkv_scale_arg = graph_utils::AddInitializer(graph, qkv_scale_initializer);
  NodeArg& qkv_zp_arg = graph_utils::AddInitializer(graph, qkv_zp_initializer);

  std::array new_tensor_node_args = {&qkv_b_arg, &qkv_scale_arg, &qkv_zp_arg};

  return new_tensor_node_args;
}
*/

Status GroupQueryAttentionFusion::ApplyImpl(
    Graph& graph,
    bool& modified,
    int graph_level,
    const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // TODO: Remove later
  for (const auto& ep : GetCompatibleExecutionProviders()) {
    std::cout << std::string(ep) << std::endl;
  }

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GroupQueryAttention", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    auto& inputs = node.MutableInputDefs();

    // Check if this fusion is called the second time.
    if (inputs[1]->Type() == nullptr) {
      continue;
    }

    std::cout << "-----------------" << std::endl;

    const ONNX_NAMESPACE::TensorProto* k_proj_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* k_scale_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* k_zero_points_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* q_proj_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* q_scale_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* q_zero_points_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* v_proj_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* v_scale_tensor = nullptr;
    const ONNX_NAMESPACE::TensorProto* v_zero_points_tensor = nullptr;

    onnxruntime::NodeArg* pos_ids_arg = nullptr;
    onnxruntime::NodeArg* cos_cache_arg = nullptr;
    onnxruntime::NodeArg* sin_cache_arg = nullptr;
    onnxruntime::NodeArg* past_key_values_key_arg = node.MutableInputDefs()[3];
    onnxruntime::NodeArg* past_key_values_value_arg = node.MutableInputDefs()[4];
    onnxruntime::NodeArg* seqlens_k = node.MutableInputDefs()[5];
    onnxruntime::NodeArg* total_seq_len = node.MutableInputDefs()[6];

    // Inputs to the newly created MatMul node
    onnxruntime::NodeArg* layer_norm = nullptr;

    std::string matMulType = "";

    Node* rotary_node_1 = nullptr;
    Node* rotary_node_2 = nullptr;
    Node* q_node = nullptr;
    Node* k_node = nullptr;
    Node* v_node = nullptr;

    for (auto* input_def : node.MutableInputDefs()) {
      // Try to find a node that produces this input
      const Node* inputNode = graph.GetProducerNode(input_def->Name());
      if (inputNode != nullptr) {
        std::cout << "Input \"" << input_def->Name() << "\" is produced by node: " << inputNode->Name() << std::endl;
        Node& rotary_or_v_node = *graph.GetNode(inputNode->Index());

        if (rotary_or_v_node.OpType() == "RotaryEmbedding") {
          if (!rotary_node_1) {
            rotary_node_1 = &rotary_or_v_node;
          } else {
            rotary_node_2 = &rotary_or_v_node;
          }

          for (auto topNodeIt = rotary_or_v_node.InputNodesBegin(); topNodeIt != rotary_or_v_node.InputNodesEnd(); ++topNodeIt) {
            // Some models might have input nodes that are unrelated to MatMulNBits or MatMul.
            if (topNodeIt->OpType() != "MatMulNBits" && topNodeIt->OpType() != "MatMul") {
              continue;
            }

            matMulType = topNodeIt->OpType();

            std::cout << "rotary input is " << (*topNodeIt).Name() << std::endl;
            auto& mat_mul_node = *graph.GetNode(topNodeIt->Index());
            std::cout << "rotary input2 is " << mat_mul_node.Name() << std::endl;
            std::cout << "rotary input3 is " << mat_mul_node.OpType() << std::endl;

            // std::cout << "Tensor1 name I am trying to load " << mat_mul_node.InputDefs()[1]->Name() << std::endl;
            // std::cout << "Tensor2 name I am trying to load " << mat_mul_node.InputDefs()[2]->Name() << std::endl;

            if (!q_node) {
              q_node = &mat_mul_node;
            } else {
              k_node = &mat_mul_node;
            }

            layer_norm = mat_mul_node.MutableInputDefs()[0];
          }

          pos_ids_arg = rotary_or_v_node.MutableInputDefs()[1];

          if (cos_cache_arg == nullptr) {
            cos_cache_arg = rotary_or_v_node.MutableInputDefs()[2];
          }

          if (sin_cache_arg == nullptr) {
            sin_cache_arg = rotary_or_v_node.MutableInputDefs()[3];
          }

          // cos and sin
        } else if (rotary_or_v_node.OpType() == "MatMulNBits" || rotary_or_v_node.OpType() == "MatMul") {
          v_node = &rotary_or_v_node;
        }
      }
    }
   
        // extract tesnors here

    if (q_proj_tensor == nullptr && !graph.GetInitializedTensor(q_node->InputDefs()[1]->Name(), q_proj_tensor)) {
      std::cout << "Unable to load Q tensor weight" << std::endl;
    }

    if (matMulType == "MatMulNBits" && q_scale_tensor == nullptr && !graph.GetInitializedTensor(q_node->InputDefs()[2]->Name(), q_scale_tensor)) {
      std::cout << "Unable to load Q scale tensor weight" << std::endl;
    }

    if (matMulType == "MatMulNBits" && q_zero_points_tensor == nullptr && !graph.GetInitializedTensor(q_node->InputDefs()[3]->Name(), q_zero_points_tensor)) {
      std::cout << "Unable to load Q zero points tensor weight" << std::endl;
    }

    if (k_proj_tensor == nullptr && !graph.GetInitializedTensor(k_node->InputDefs()[1]->Name(), k_proj_tensor)) {
      std::cout << "Unable to load K tensor weight" << std::endl;
    }

    if (matMulType == "MatMulNBits" && k_scale_tensor == nullptr && !graph.GetInitializedTensor(k_node->InputDefs()[2]->Name(), k_scale_tensor)) {
      std::cout << "Unable to load K scale tensor weight" << std::endl;
    }

    if (matMulType == "MatMulNBits" && k_zero_points_tensor == nullptr && !graph.GetInitializedTensor(k_node->InputDefs()[3]->Name(), k_zero_points_tensor)) {
      std::cout << "Unable to load K zero points tensor weight" << std::endl;
    }

    if (v_proj_tensor == nullptr && !graph.GetInitializedTensor(v_node->InputDefs()[1]->Name(), v_proj_tensor)) {
      std::cout << "Unable to load V tensor weight" << std::endl;
    }

    if (matMulType == "MatMulNBits" && v_scale_tensor == nullptr && !graph.GetInitializedTensor(v_node->InputDefs()[2]->Name(), v_scale_tensor)) {
      std::cout << "Unable to load V scale tensor weight" << std::endl;
    }

    if (matMulType == "MatMulNBits" && v_zero_points_tensor == nullptr && !graph.GetInitializedTensor(v_node->InputDefs()[3]->Name(), v_zero_points_tensor)) {
      std::cout << "Unable to load V zero points tensor weight" << std::endl;
    }

    std::cout << "K tensor Dimensions: ";
    for (int i = 0; i < k_proj_tensor->dims_size(); ++i) {
      std::cout << k_proj_tensor->dims(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Q tensor Dimensions: ";
    for (int i = 0; i < q_proj_tensor->dims_size(); ++i) {
      std::cout << q_proj_tensor->dims(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "V tensor Dimensions: ";
    for (int i = 0; i < v_proj_tensor->dims_size(); ++i) {
      std::cout << v_proj_tensor->dims(i) << " ";
    }

    [[maybe_unused]] int64_t hidden_size = q_proj_tensor->dims(0);

    //optimizer_utils::MergeMatMulWeights(q)

    [[maybe_unused]]auto& aa = MergeQkvWeights2(graph, q_proj_tensor, k_proj_tensor, v_proj_tensor);
    //auto qkv_args = MergeQkvWeights(graph, hidden_size, q_proj_tensor->dims(1), q_proj_tensor->dims(2), q_proj_tensor, k_proj_tensor, v_proj_tensor, q_scale_tensor, q_zero_points_tensor, k_scale_tensor, k_zero_points_tensor, v_scale_tensor, v_zero_points_tensor);

    // todo prob remove this
    if (pos_ids_arg == nullptr) {
      std::cout << "Position IDs argument is not available; cannot fuse GroupQueryAttention." << std::endl;
      continue;
    }

    // Inputs we have for mat mul are layer norm and merged/packed QKV weights.
    //const std::array mmnb_input_defs{layer_norm, qkv_args[0], qkv_args[1], qkv_args[2]};
    const std::array mmnb_input_defs2{layer_norm, &aa};

    const onnx::TypeProto* layer_norm_tensor_proto = layer_norm->TypeAsProto();
    onnx::TypeProto mutable_mat_mul_tensor_proto = *layer_norm_tensor_proto;
    auto* tensor_type = mutable_mat_mul_tensor_proto.mutable_tensor_type();
    auto* shape = tensor_type->mutable_shape();

    // Ensure the shape has at least 3 dimensions
    if (shape->dim_size() > 2) {
      auto* third_dim = shape->mutable_dim(2);
      //third_dim->set_dim_value(3 * hidden_size);
      third_dim->set_dim_value(4096);
    }

    //auto& matmul_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MatMul_output"), &mutable_mat_mul_tensor_proto);
    auto& matmul_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MatMul_output"), &mutable_mat_mul_tensor_proto);

    const std::array mmnb_output_defs{&matmul_output};

    // Add MatMulNBits
    /*
    Node& mat_mul_n_bits_new_node = graph.AddNode(graph.GenerateNodeName("MatMulNBits"),
                                                  "MatMulNBits",
                                                  "MatMulNBits fusion node",
                                                  mmnb_input_defs2,
                                                  mmnb_output_defs,
                                                  &q_node->GetAttributes(),
                                                  kMSDomain);
                                                  */

        Node& mat_mul_n_bits_new_node = graph.AddNode(graph.GenerateNodeName("MatMul"),
                                                  "MatMul",
                                                  "MatMul fusion node",
                                                  mmnb_input_defs2,
                                                  mmnb_output_defs,
                                                  &q_node->GetAttributes(),
                                                  kOnnxDomainAlias);

    //AttachNodeAttribute(mat_mul_n_bits_new_node, "N", 3 * mat_mul_n_bits_new_node.GetAttributes().at("N").i(), AttributeProto_AttributeType_INT);

    mat_mul_n_bits_new_node.SetExecutionProviderType(node.GetExecutionProviderType());

    graph_utils::FinalizeNodeFusion(graph, {*q_node, *k_node, *v_node, *rotary_node_1, *rotary_node_2}, mat_mul_n_bits_new_node);

    // Make sure the matmulnbits node has correct output defs
    auto& mat_mut_output_defs = mat_mul_n_bits_new_node.MutableOutputDefs();
    mat_mut_output_defs.assign(mmnb_output_defs.begin(), mmnb_output_defs.end());

    [[maybe_unused]] const onnxruntime::Node* producer = graph.GetProducerNode(matmul_output.Name());

    AttachNodeAttribute(node, "do_rotary", 1, AttributeProto_AttributeType_INT);

    std::string empty_name;
    auto& emptyNode = graph.GetOrCreateNodeArg(empty_name, nullptr);

    const std::array gqa_input_defs{
        &matmul_output,
        &emptyNode,
        &emptyNode,
        past_key_values_key_arg,
        past_key_values_value_arg,
        seqlens_k,
        total_seq_len,
        cos_cache_arg,
        sin_cache_arg,
        pos_ids_arg};

    auto& gqaInputArgs = node.MutableInputArgsCount();
    gqaInputArgs[7] = 1;
    gqaInputArgs[8] = 1;
    gqaInputArgs[9] = 1;

    auto& input_defs = node.MutableInputDefs();
    input_defs.assign(gqa_input_defs.begin(), gqa_input_defs.end());

    [[maybe_unused]] NodeAttributes node_attributes2 = node.GetAttributes();

    ORT_RETURN_IF_ERROR(graph.Resolve());

    modified = true;

    [[maybe_unused]] const onnxruntime::Node* producer44 = graph.GetProducerNode(matmul_output.Name());

    // graph_utils::FinalizeNodeFusion(graph, {node}, node);

    [[maybe_unused]] int sodjsapidjad = 1;
    [[maybe_unused]] int sodjsapidjad2 = 1;

    [[maybe_unused]] int sodjsapidjad345 = 1;
  }

  return Status::OK();
}
}  // namespace onnxruntime
