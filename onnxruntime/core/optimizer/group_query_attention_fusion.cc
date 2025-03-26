// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/group_query_attention_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

template <typename T, typename OutT>
void DequantizePreGqaWeights(size_t M, size_t K, size_t N, const T* input,
                const OutT* scale, OutT* output, const T* zero_point) {
  for (size_t m = 0; m < M; m++) {
    for (size_t k = 0; k < K; k++) {
      size_t index = K * m + k;
      auto zp = zero_point ? static_cast<int32_t>(zero_point[index / 2]) : 0;
      auto sc = static_cast<float>(scale[index]);
      for (size_t n = 0; n < N; n++) {
        *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc);
      }
    }
  }
}

NodeArg& MergeQkvWeights(Graph& graph,
                         int64_t input_dim,       // For example, 3072
                         int64_t qkv_second_dim,  // Number of query heads, e.g., 24
                         int64_t qkv_third_dim,   // Size per head, e.g., 64
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

  Initializer q_scale_tensor_initializer(*q_scale_tensor, graph.ModelPath());
  Initializer q_zero_tensor_initializer(*q_zero_point_tensor, graph.ModelPath());

  Initializer k_scale_tensor_initializer(*k_scale_tensor, graph.ModelPath());
  Initializer k_zero_tensor_initializer(*k_zero_point_tensor, graph.ModelPath());

  Initializer v_scale_tensor_initializer(*v_scale_tensor, graph.ModelPath());
  Initializer v_zero_tensor_initializer(*v_zero_point_tensor, graph.ModelPath());

  auto data_type = q_tensor->data_type();

  int64_t single_tensor_output_dim = qkv_second_dim * qkv_third_dim; 
  int64_t total_output_dim = 3 * single_tensor_output_dim; // Because we have 3 tensors to merge.
  
  // ----- Use the data() API to retrieve the tensor data -----
  // Get the number of elements (each element is a uint8_t).
  size_t q_elements = q_initializer.size();
  size_t k_elements = k_initializer.size();
  size_t v_elements = v_initializer.size();

  // Get pointers to the underlying data.
  const uint8_t* q_data = q_initializer.data<uint8_t>();
  const uint8_t* k_data = k_initializer.data<uint8_t>();
  const uint8_t* v_data = v_initializer.data<uint8_t>();

  const MLFloat16* q_scale_data = q_scale_tensor_initializer.data<MLFloat16>();
  const uint8_t* q_zero_points_data = q_zero_tensor_initializer.data<uint8_t>();

  const MLFloat16* k_scale_data = k_scale_tensor_initializer.data<MLFloat16>();
  const uint8_t* k_zero_points_data = k_zero_tensor_initializer.data<uint8_t>();

  const MLFloat16* v_scale_data = v_scale_tensor_initializer.data<MLFloat16>();
  const uint8_t* v_zero_points_data = v_zero_tensor_initializer.data<uint8_t>();

  std::vector<uint8_t> merged_qkv_data;
  size_t element_count = q_elements + k_elements + v_elements;
  merged_qkv_data.reserve(element_count);

  std::vector<MLFloat16> merged_qkv_scale_data;
  merged_qkv_scale_data.reserve(input_dim * qkv_second_dim);

  std::vector<uint8_t> merged_qkv_zero_points_data;
  merged_qkv_zero_points_data.reserve(input_dim * qkv_second_dim / 2);

    /*
  std::vector<MLFloat16> q_dequantized_data;
  q_dequantized_data.resize(element_count);

  std::vector<MLFloat16> k_dequantized_data;
  k_dequantized_data.resize(element_count);

  std::vector<MLFloat16> v_dequantized_data;
  v_dequantized_data.resize(element_count);

  DequantizePreGqaWeights<uint8_t, MLFloat16>(input_dim, qkv_second_dim, qkv_third_dim, q_data, q_scale_data, q_dequantized_data.data(), q_zero_points_data);
  DequantizePreGqaWeights<uint8_t, MLFloat16>(input_dim, qkv_second_dim, qkv_third_dim, k_data, k_scale_data, k_dequantized_data.data(), k_zero_points_data);
  DequantizePreGqaWeights<uint8_t, MLFloat16>(input_dim, qkv_second_dim, qkv_third_dim, v_data, v_scale_data, v_dequantized_data.data(), v_zero_points_data);
  */
  optimizer_utils::MergeMatMulWeights<uint8_t>(q_data, k_data, v_data, merged_qkv_data, input_dim, single_tensor_output_dim);
  optimizer_utils::MergeMatMulWeights<MLFloat16>(q_scale_data, k_scale_data, v_scale_data, merged_qkv_scale_data, input_dim, qkv_second_dim);
  optimizer_utils::MergeMatMulWeights<uint8_t>(q_zero_points_data, k_zero_points_data, v_zero_points_data, merged_qkv_zero_points_data, input_dim, qkv_second_dim / 2);

  ONNX_NAMESPACE::TensorProto merged_qkv_initializer;
  merged_qkv_initializer.set_name(graph.GenerateNodeArgName("qkv_weights"));

  merged_qkv_initializer.add_dims(input_dim);
  merged_qkv_initializer.add_dims(total_output_dim);
  merged_qkv_initializer.set_data_type(data_type);


  ONNX_NAMESPACE::TensorProto merged_qkv_scale_initializer;
  merged_qkv_scale_initializer.set_name(graph.GenerateNodeArgName("qkv_weights_scale"));

  merged_qkv_scale_initializer.add_dims(input_dim);
  merged_qkv_scale_initializer.add_dims(qkv_second_dim);
  merged_qkv_scale_initializer.set_data_type(q_scale_tensor->data_type());

  ONNX_NAMESPACE::TensorProto merged_qkv_zp_initializer;
  merged_qkv_zp_initializer.set_name(graph.GenerateNodeArgName("qkv_weights_zp"));

  merged_qkv_zp_initializer.add_dims(input_dim);
  merged_qkv_zp_initializer.add_dims(qkv_second_dim / 2);
  merged_qkv_zp_initializer.set_data_type(data_type);

  utils::SetRawDataInTensorProto(merged_qkv_initializer, merged_qkv_data.data(), gsl::narrow<size_t>(element_count) * sizeof(uint8_t));
  utils::SetRawDataInTensorProto(merged_qkv_scale_initializer, merged_qkv_scale_data.data(), gsl::narrow<size_t>(input_dim * qkv_second_dim) * sizeof(MLFloat16));
  utils::SetRawDataInTensorProto(merged_qkv_zp_initializer, merged_qkv_zero_points_data.data(), gsl::narrow<size_t>(input_dim * qkv_second_dim / 2) * sizeof(uint8_t));

    [[maybe_unused]]  NodeArg& a1 = graph_utils::AddInitializer(graph, merged_qkv_scale_initializer);
  [[maybe_unused]]  NodeArg& a2 = graph_utils::AddInitializer(graph, merged_qkv_zp_initializer);

  return graph_utils::AddInitializer(graph, merged_qkv_initializer);
}

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

    std::cout << inputs.size() << std::endl;

    for (auto input : inputs) {
      std::cout << input->Name() << std::endl;
      std::cout << *input->Type() << std::endl;
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
    onnxruntime::NodeArg* past_key_values_key_arg = nullptr;
    onnxruntime::NodeArg* past_key_values_value_arg = nullptr;
    onnxruntime::NodeArg* seqlens_k = nullptr;
    onnxruntime::NodeArg* total_seq_len = nullptr;

    // Inputs to the newly created MatMul node
    onnxruntime::NodeArg* layer_norm = nullptr;

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

          for (auto inner = rotary_or_v_node.InputNodesBegin(); inner != rotary_or_v_node.InputNodesEnd(); ++inner) {
            std::cout << "rotary input is " << (*inner).Name() << std::endl;
            auto& mat_mul_node = *graph.GetNode(inner->Index());
            std::cout << "rotary input2 is " << mat_mul_node.Name() << std::endl;
            std::cout << "rotary input3 is " << mat_mul_node.OpType() << std::endl;

            std::cout << "Tensor1 name I am trying to load " << mat_mul_node.InputDefs()[1]->Name() << std::endl;
            std::cout << "Tensor2 name I am trying to load " << mat_mul_node.InputDefs()[2]->Name() << std::endl;

            if (!k_node) {
              k_node = &mat_mul_node;
            } else {
              q_node = &mat_mul_node;
            }

            layer_norm = mat_mul_node.MutableInputDefs()[0];

            for (auto input_of_mat_mul_node : mat_mul_node.MutableInputDefs()) {
              std::cout << "input of mat mul node " << input_of_mat_mul_node->Name() << std::endl; 
            }

            if (k_proj_tensor == nullptr && !graph.GetInitializedTensor(mat_mul_node.InputDefs()[1]->Name(), k_proj_tensor)) {
              std::cout << "Unable to load K tensor weight" << std::endl;
            }

            if (k_scale_tensor == nullptr && !graph.GetInitializedTensor(mat_mul_node.InputDefs()[2]->Name(), k_scale_tensor)) {
              std::cout << "Unable to load K scale tensor weight" << std::endl;
            }

            if (k_zero_points_tensor == nullptr && !graph.GetInitializedTensor(mat_mul_node.InputDefs()[3]->Name(), k_zero_points_tensor)) {
              std::cout << "Unable to load K zero points tensor weight" << std::endl;
            }

            if (q_proj_tensor == nullptr && !graph.GetInitializedTensor(mat_mul_node.InputDefs()[1]->Name(), q_proj_tensor)) {
              std::cout << "Unable to load Q tensor weight" << std::endl;
            }

            if (q_scale_tensor == nullptr && !graph.GetInitializedTensor(mat_mul_node.InputDefs()[2]->Name(), q_scale_tensor)) {
              std::cout << "Unable to load Q scale tensor weight" << std::endl;
            }

            if (q_zero_points_tensor == nullptr && !graph.GetInitializedTensor(mat_mul_node.InputDefs()[3]->Name(), q_zero_points_tensor)) {
              std::cout << "Unable to load Q zero points tensor weight" << std::endl;
            }
          }

          pos_ids_arg = rotary_or_v_node.MutableInputDefs()[1];

          if (cos_cache_arg == nullptr) {
            cos_cache_arg = rotary_or_v_node.MutableInputDefs()[2];
          }

          if (sin_cache_arg == nullptr) {
            sin_cache_arg = rotary_or_v_node.MutableInputDefs()[3];
          }

          // cos and sin
        } else if (rotary_or_v_node.OpType() == "MatMulNBits") {
          v_node = &rotary_or_v_node;
          if (v_proj_tensor == nullptr && !graph.GetInitializedTensor(rotary_or_v_node.InputDefs()[1]->Name(), v_proj_tensor)) {
            std::cout << "Unable to load V tensor weight" << std::endl;
          }

          if (v_scale_tensor == nullptr && !graph.GetInitializedTensor(rotary_or_v_node.InputDefs()[2]->Name(), v_scale_tensor)) {
            std::cout << "Unable to load V scale tensor weight" << std::endl;
          }

          if (v_zero_points_tensor == nullptr && !graph.GetInitializedTensor(rotary_or_v_node.InputDefs()[3]->Name(), v_zero_points_tensor)) {
            std::cout << "Unable to load V zero points tensor weight" << std::endl;
          }
        }

        if (input_def->Name() == "seqlens_k") {
          seqlens_k = input_def;
        } else if (input_def->Name() == "total_seq_len") {
          total_seq_len = input_def;
        }

      } else {
        std::cout << "Input \"" << input_def->Name() << "\" is not produced by any node (it is likely an initializer or constant)" << std::endl;

        if (input_def->Name().find(".key") != std::string::npos) {
          past_key_values_key_arg = input_def;

        } else if (input_def->Name().find(".value") != std::string::npos) {
          past_key_values_value_arg = input_def;
        }
      }
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

    int64_t hidden_size = q_proj_tensor->dims(0);

    onnxruntime::NodeArg& fused_qkv = MergeQkvWeights(graph, hidden_size, q_proj_tensor->dims(1), q_proj_tensor->dims(2), q_proj_tensor, k_proj_tensor, v_proj_tensor, q_scale_tensor, q_zero_points_tensor, k_scale_tensor, k_zero_points_tensor, v_scale_tensor, v_zero_points_tensor);

    if (pos_ids_arg == nullptr) {
      std::cout << "Position IDs argument is not available; cannot fuse GroupQueryAttention." << std::endl;
      continue;
    }

    [[maybe_unused]] const std::array gqa_input_defs{pos_ids_arg, &fused_qkv, cos_cache_arg, sin_cache_arg, past_key_values_key_arg, past_key_values_value_arg, seqlens_k, total_seq_len};

    NodeAttributes node_attributes = node.GetAttributes();
    ONNX_NAMESPACE::AttributeProto attr;
    attr.set_name("do_rotary");
    attr.set_i(1);
    node_attributes["do_rotary"] = attr;


    // Other inputs here are B, Scale and zero points
        [[maybe_unused]] const std::array mmnb_input_defs{layer_norm};

        NodeAttributes mmnb_node_atributes = q_node->GetAttributes();
        ONNX_NAMESPACE::AttributeProto mmnb_N_attr_proto;
        mmnb_N_attr_proto.set_name("N");
        mmnb_N_attr_proto.set_i(3 * mmnb_node_atributes["N"].i());
        mmnb_node_atributes["N"] = mmnb_N_attr_proto;

    // Add MatMulNBits
        [[maybe_unused]] Node& mat_mul_n_bits_new_node = graph.AddNode(graph.GenerateNodeName("MatMulNBits"),
                                   "MatMulNBits",
                                   "MatMulNBits fused node",
                                   mmnb_input_defs,
                                   {},
                                   &mmnb_node_atributes,
                                   kMSDomain);

    // Now add the fused GroupQueryAttention node.
    Node& gqa_node = graph.AddNode(graph.GenerateNodeName("GroupQueryAttention"),
                                   "GroupQueryAttention",
                                   "Fused GroupQueryAttention subgraphs",
                                   gqa_input_defs,
                                   {},
                                   &node_attributes,
                                   kMSDomain);

    mat_mul_n_bits_new_node.MutableOutputDefs().push_back(graph.GetNodeArg(gqa_node.Name()));

    gqa_node.SetExecutionProviderType(node.GetExecutionProviderType());

    graph_utils::FinalizeNodeFusion(graph, {*q_node, *k_node, *v_node, *rotary_node_1, *rotary_node_2}, gqa_node);

    [[maybe_unused]] int sodjsapidjad = 1;
    [[maybe_unused]] int sodjsapidjad2 = 1;

    [[maybe_unused]] int sodjsapidjad345 = 1;
  }

  return Status::OK();
}
}  // namespace onnxruntime
