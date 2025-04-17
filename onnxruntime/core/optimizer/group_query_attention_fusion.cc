// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/group_query_attention_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include <onnx/defs/attr_proto_util.h>

#define DEBUG_LOG(x) LOGS(logger, VERBOSE) << x

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

static NodeArg& MergeQkvWeightsForMatMul(Graph& graph,
                                         int64_t q_hidden_size,
                                         int64_t kv_hidden_size,
                                         const TensorProto* q_tensor,
                                         const TensorProto* k_tensor,
                                         const TensorProto* v_tensor) {
  assert(nullptr != q_tensor);
  assert(nullptr != k_tensor);
  assert(nullptr != v_tensor);

  int64_t input_hidden_size = q_tensor->dims(0);

  Initializer q_initializer(*q_tensor, graph.ModelPath());
  Initializer k_initializer(*k_tensor, graph.ModelPath());
  Initializer v_initializer(*v_tensor, graph.ModelPath());

  int64_t output_hidden_size = q_hidden_size + 2 * kv_hidden_size;

  TensorProto qkv_weight_initializer;
  qkv_weight_initializer.set_name(graph.GenerateNodeArgName("qkv_weight"));
  qkv_weight_initializer.add_dims(input_hidden_size);
  qkv_weight_initializer.add_dims(output_hidden_size);
  qkv_weight_initializer.set_data_type(q_tensor->data_type());

  const MLFloat16* q_data = q_initializer.data<MLFloat16>();
  const MLFloat16* k_data = k_initializer.data<MLFloat16>();
  const MLFloat16* v_data = v_initializer.data<MLFloat16>();

  int64_t element_count = input_hidden_size * output_hidden_size;
  std::vector<MLFloat16> merged_qkv_weight;
  merged_qkv_weight.reserve(gsl::narrow<size_t>(element_count));

  optimizer_utils::MergeMatMulWeightsByRow(q_data, k_data, v_data, merged_qkv_weight, input_hidden_size, q_hidden_size, kv_hidden_size);
  utils::SetRawDataInTensorProto(qkv_weight_initializer, merged_qkv_weight.data(), element_count * sizeof(MLFloat16));

  return graph_utils::AddInitializer(graph, qkv_weight_initializer);
}

static std::vector<NodeArg*> MergeQkvWeightsForMatMulNBits(
    Graph& graph,
    int64_t q_hidden_size,
    int64_t kv_hidden_size,
    int64_t blocks,
    int64_t block_size,
    const TensorProto* q_tensor,
    const TensorProto* k_tensor,
    const TensorProto* v_tensor,
    const TensorProto* q_scale_tensor,
    const TensorProto* q_zero_point_tensor,
    const TensorProto* k_scale_tensor,
    const TensorProto* k_zero_point_tensor,
    const TensorProto* v_scale_tensor,
    const TensorProto* v_zero_point_tensor) {
  // B and scale tensors are required.
  assert(q_tensor != nullptr);
  assert(k_tensor != nullptr);
  assert(v_tensor != nullptr);
  assert(q_scale_tensor != nullptr);
  assert(k_scale_tensor != nullptr);
  assert(v_scale_tensor != nullptr);

  // Determine if all zero-point tensors exist.
  bool has_zero_points = (q_zero_point_tensor != nullptr &&
                          k_zero_point_tensor != nullptr &&
                          v_zero_point_tensor != nullptr);

  Initializer q_initializer(*q_tensor, graph.ModelPath());
  Initializer k_initializer(*k_tensor, graph.ModelPath());
  Initializer v_initializer(*v_tensor, graph.ModelPath());

  Initializer q_scale_initializer(*q_scale_tensor, graph.ModelPath());
  Initializer k_scale_initializer(*k_scale_tensor, graph.ModelPath());
  Initializer v_scale_initializer(*v_scale_tensor, graph.ModelPath());

  const uint8_t* q_data = q_initializer.data<uint8_t>();
  const uint8_t* k_data = k_initializer.data<uint8_t>();
  const uint8_t* v_data = v_initializer.data<uint8_t>();

  const MLFloat16* q_scale_data = q_scale_initializer.data<MLFloat16>();
  const MLFloat16* k_scale_data = k_scale_initializer.data<MLFloat16>();
  const MLFloat16* v_scale_data = v_scale_initializer.data<MLFloat16>();

  int64_t output_hidden_size = q_hidden_size + 2 * kv_hidden_size;

  TensorProto qkv_weight_initializer;
  qkv_weight_initializer.set_name(graph.GenerateNodeArgName("qkv_weight"));
  qkv_weight_initializer.add_dims(output_hidden_size);
  qkv_weight_initializer.add_dims(blocks);
  qkv_weight_initializer.add_dims(block_size);
  qkv_weight_initializer.set_data_type(q_tensor->data_type());

  TensorProto qkv_scale_initializer;
  qkv_scale_initializer.set_name(graph.GenerateNodeArgName("qkv_scale"));

  // Preserve scale tensor shape (the dimension is either 1 or 2).
  if (q_scale_tensor->dims().size() > 1) {
    qkv_scale_initializer.add_dims(output_hidden_size);
    qkv_scale_initializer.add_dims(blocks);
  } else {
    qkv_scale_initializer.add_dims(output_hidden_size * blocks);
  }
  qkv_scale_initializer.set_data_type(q_scale_tensor->data_type());

  int64_t element_count = output_hidden_size * blocks * block_size;
  std::vector<uint8_t> merged_qkv_weight;
  merged_qkv_weight.reserve(gsl::narrow<size_t>(element_count));

  int64_t scale_elements_count = output_hidden_size * blocks;
  std::vector<MLFloat16> merged_qkv_scale;
  merged_qkv_scale.reserve(gsl::narrow<size_t>(scale_elements_count));

  optimizer_utils::MergeMatMulWeightsByBlocks(q_data, k_data, v_data, merged_qkv_weight, q_hidden_size, kv_hidden_size, blocks, block_size);
  optimizer_utils::MergeMatMulWeightsByBlocks(q_scale_data, k_scale_data, v_scale_data, merged_qkv_scale, q_hidden_size, kv_hidden_size, blocks, 1);

  utils::SetRawDataInTensorProto(qkv_weight_initializer, merged_qkv_weight.data(), element_count * sizeof(uint8_t));
  utils::SetRawDataInTensorProto(qkv_scale_initializer, merged_qkv_scale.data(), scale_elements_count * sizeof(MLFloat16));

  NodeArg& qkv_weight_arg = graph_utils::AddInitializer(graph, qkv_weight_initializer);
  NodeArg& qkv_scale_arg = graph_utils::AddInitializer(graph, qkv_scale_initializer);

  std::vector<NodeArg*> result_node_args = {&qkv_weight_arg, &qkv_scale_arg};

  if (has_zero_points) {
    Initializer q_zp_initializer(*q_zero_point_tensor, graph.ModelPath());
    Initializer k_zp_initializer(*k_zero_point_tensor, graph.ModelPath());
    Initializer v_zp_initializer(*v_zero_point_tensor, graph.ModelPath());

    const uint8_t* q_zero_points_data = q_zp_initializer.data<uint8_t>();
    const uint8_t* k_zero_points_data = k_zp_initializer.data<uint8_t>();
    const uint8_t* v_zero_points_data = v_zp_initializer.data<uint8_t>();

    TensorProto qkv_zp_initializer;

    // We use 4 bit quantization, hence dividing by 2 since we need 1/2 of the bytes.
    int64_t zp_elements_count = output_hidden_size * blocks / 2;

    qkv_zp_initializer.set_name(graph.GenerateNodeArgName("qkv_zp"));
    qkv_zp_initializer.add_dims(zp_elements_count);
    qkv_zp_initializer.set_data_type(q_zero_point_tensor->data_type());

    std::vector<uint8_t> merged_qkv_zp;
    merged_qkv_zp.reserve(gsl::narrow<size_t>(zp_elements_count));

    optimizer_utils::MergeMatMulWeightsByBlocks(q_zero_points_data, k_zero_points_data, v_zero_points_data,
                                                merged_qkv_zp, q_hidden_size, kv_hidden_size, blocks / 2, 1);

    utils::SetRawDataInTensorProto(qkv_zp_initializer, merged_qkv_zp.data(), zp_elements_count * sizeof(uint8_t));

    NodeArg& qkv_zp_arg = graph_utils::AddInitializer(graph, qkv_zp_initializer);
    result_node_args.push_back(&qkv_zp_arg);
  }

  return result_node_args;
}

static bool LoadQKVProjectionTensors(Graph& graph,
                                     bool quantization_used,
                                     Node* q_node,
                                     Node* k_node,
                                     Node* v_node,
                                     const TensorProto*& q_proj_tensor,
                                     const TensorProto*& k_proj_tensor,
                                     const TensorProto*& v_proj_tensor,
                                     const TensorProto*& q_scale_tensor,
                                     const TensorProto*& k_scale_tensor,
                                     const TensorProto*& v_scale_tensor,
                                     const TensorProto*& q_zero_points_tensor,
                                     const TensorProto*& k_zero_points_tensor,
                                     const TensorProto*& v_zero_points_tensor) {
  // Only support bits = 4 fusion on MatMulNBits.
  if (quantization_used && (q_node->GetAttributes().at("bits").i() != 4 || k_node->GetAttributes().at("bits").i() != 4 || v_node->GetAttributes().at("bits").i() != 4)) {
    return false;
  }

  if (!graph.GetInitializedTensor(q_node->InputDefs()[1]->Name(), q_proj_tensor)) {
    return false;
  }

  if (quantization_used && !graph.GetInitializedTensor(q_node->InputDefs()[2]->Name(), q_scale_tensor)) {
    return false;
  }

  if (!graph.GetInitializedTensor(k_node->InputDefs()[1]->Name(), k_proj_tensor)) {
    return false;
  }

  if (quantization_used && !graph.GetInitializedTensor(k_node->InputDefs()[2]->Name(), k_scale_tensor)) {
    return false;
  }

  if (!graph.GetInitializedTensor(v_node->InputDefs()[1]->Name(), v_proj_tensor)) {
    return false;
  }

  if (quantization_used && !graph.GetInitializedTensor(v_node->InputDefs()[2]->Name(), v_scale_tensor)) {
    return false;
  }

  // Extract zero points tensors only if they're present.
  if (quantization_used && q_node->InputDefs().size() > 3 &&
      !graph.GetInitializedTensor(q_node->InputDefs()[3]->Name(), q_zero_points_tensor)) {
    return false;
  }

  if (quantization_used && k_node->InputDefs().size() > 3 &&
      !graph.GetInitializedTensor(k_node->InputDefs()[3]->Name(), k_zero_points_tensor)) {
    return false;
  }

  if (quantization_used && v_node->InputDefs().size() > 3 &&
      !graph.GetInitializedTensor(v_node->InputDefs()[3]->Name(), v_zero_points_tensor)) {
    return false;
  }

  if (quantization_used) {
    if ((q_zero_points_tensor || k_zero_points_tensor || v_zero_points_tensor) &&
        (!q_zero_points_tensor || !k_zero_points_tensor || !v_zero_points_tensor)) {
      return false;
    }

    // Support only packed zp tensors for now.
    if (q_zero_points_tensor && k_zero_points_tensor && v_zero_points_tensor && (q_zero_points_tensor->data_type() != TensorProto::UINT8 || k_zero_points_tensor->data_type() != TensorProto::UINT8 || v_zero_points_tensor->data_type() != TensorProto::UINT8)) {
      return false;
    }

    return q_proj_tensor->data_type() == TensorProto::UINT8 && k_proj_tensor->data_type() == TensorProto::UINT8 && v_proj_tensor->data_type() == TensorProto::UINT8 &&
           q_scale_tensor->data_type() == TensorProto::FLOAT16 && k_scale_tensor->data_type() == TensorProto::FLOAT16 && v_scale_tensor->data_type() == TensorProto::FLOAT16;
  } else {
    return q_proj_tensor->data_type() == TensorProto::FLOAT16 && k_proj_tensor->data_type() == TensorProto::FLOAT16 && v_proj_tensor->data_type() == TensorProto::FLOAT16;
  }
}

static bool CheckIfAnyOfRequiredGQANodesDoesNotExist(Node* rotary_node_1, Node* rotary_node_2, Node* q_node, Node* k_node, Node* v_node) {
  return rotary_node_1 == nullptr || rotary_node_2 == nullptr || q_node == nullptr || k_node == nullptr || v_node == nullptr;
}

static void FusePreGQANodes(Graph& graph, Node* q_node, Node* k_node, Node* v_node, Node* rotary_node_1, Node* rotary_node_2, Node* new_node, NodeArg& new_node_output_arg) {
  graph_utils::MoveAllNodeInputEdges(graph, *q_node, *new_node);

  auto target_idx = new_node->Index();

  // Get and remove the old output edges.
  auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(*rotary_node_2);
  graph_utils::GraphEdge::RemoveGraphEdges(graph, output_edges);

  // Add the new output edges to the new node.
  for (auto cur = output_edges.cbegin(), end = output_edges.cend(); cur != end; ++cur) {
    graph.AddEdge(target_idx, cur->dst_node, cur->src_arg_index, cur->dst_arg_index);
  }

  auto nodes = {q_node, k_node, v_node, rotary_node_1, rotary_node_2};

  // Remove old nodes and their outdoing edges.
  for (Node* node : nodes) {
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  const std::array output_defs{&new_node_output_arg};

  auto& new_node_output_defs = new_node->MutableOutputDefs();
  new_node_output_defs.assign(output_defs.begin(), output_defs.end());
}

Status GroupQueryAttentionFusion::ApplyImpl(
    Graph& graph,
    bool& modified,
    int graph_level,
    const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

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

    const auto& gqa_node_attrs = node.GetAttributes();
    const auto do_rotary_attr_it = gqa_node_attrs.find("do_rotary");

    // Check if GQA is already fused.
    if (do_rotary_attr_it != gqa_node_attrs.end() && do_rotary_attr_it->second.i() == 1) {
      continue;
    }

    const TensorProto* k_proj_tensor = nullptr;
    const TensorProto* k_scale_tensor = nullptr;
    const TensorProto* k_zero_points_tensor = nullptr;
    const TensorProto* q_proj_tensor = nullptr;
    const TensorProto* q_scale_tensor = nullptr;
    const TensorProto* q_zero_points_tensor = nullptr;
    const TensorProto* v_proj_tensor = nullptr;
    const TensorProto* v_scale_tensor = nullptr;
    const TensorProto* v_zero_points_tensor = nullptr;

    NodeArg* cos_cache_arg = nullptr;
    NodeArg* sin_cache_arg = nullptr;
    NodeArg* past_key_values_key_arg = node.MutableInputDefs()[3];
    NodeArg* past_key_values_value_arg = node.MutableInputDefs()[4];
    NodeArg* seqlens_k = node.MutableInputDefs()[5];
    NodeArg* total_seq_len = node.MutableInputDefs()[6];

    bool quantization_used = false;

    Node* rotary_node_1 = nullptr;
    Node* rotary_node_2 = nullptr;
    Node* q_node = nullptr;
    Node* k_node = nullptr;
    Node* v_node = nullptr;

    for (auto pre_gqa_node = node.InputNodesBegin(); pre_gqa_node != node.InputNodesEnd(); ++pre_gqa_node) {
      Node& rotary_or_v_node = *graph.GetNode(pre_gqa_node->Index());

      if (rotary_or_v_node.OpType() == "RotaryEmbedding") {
        if (!rotary_node_1) {
          rotary_node_1 = &rotary_or_v_node;
        } else {
          rotary_node_2 = &rotary_or_v_node;
        }

        for (auto pre_rotary_node = rotary_or_v_node.InputNodesBegin(); pre_rotary_node != rotary_or_v_node.InputNodesEnd(); ++pre_rotary_node) {
          // Some models might have input nodes that are unrelated to MatMulNBits or MatMul.
          if (pre_rotary_node->OpType() != "MatMulNBits" && pre_rotary_node->OpType() != "MatMul") {
            continue;
          }

          auto& mat_mul_or_nbits_node = *graph.GetNode(pre_rotary_node->Index());

          // Q always comes before K.
          if (!q_node) {
            q_node = &mat_mul_or_nbits_node;
          } else {
            k_node = &mat_mul_or_nbits_node;
          }
        }

        if (cos_cache_arg == nullptr) {
          cos_cache_arg = rotary_or_v_node.MutableInputDefs()[2];
        }

        if (sin_cache_arg == nullptr) {
          sin_cache_arg = rotary_or_v_node.MutableInputDefs()[3];
        }
      } else if (rotary_or_v_node.OpType() == "MatMulNBits" || rotary_or_v_node.OpType() == "MatMul") {
        v_node = &rotary_or_v_node;
      }
    }

    if (CheckIfAnyOfRequiredGQANodesDoesNotExist(rotary_node_1, rotary_node_2, q_node, k_node, v_node)) {
      // Some of the required pre-GQA nodes required for fusion were not retrieved,
      // this can be expected if the model has extra nodes in between MatMuls and rotary embeddings.
      continue;
    }

    if (q_node->OpType() == "MatMulNBits" && k_node->OpType() == "MatMulNBits" && v_node->OpType() == "MatMulNBits") {
      quantization_used = true;
    } else if (q_node->OpType() == "MatMul" && k_node->OpType() == "MatMul" && v_node->OpType() == "MatMul") {
      quantization_used = false;
    } else {
      continue;
    }

    if (!LoadQKVProjectionTensors(graph,
                                  quantization_used,
                                  q_node,
                                  k_node,
                                  v_node,
                                  q_proj_tensor,
                                  k_proj_tensor,
                                  v_proj_tensor,
                                  q_scale_tensor,
                                  k_scale_tensor,
                                  v_scale_tensor,
                                  q_zero_points_tensor,
                                  k_zero_points_tensor,
                                  v_zero_points_tensor)) {
      DEBUG_LOG("Some of the required tensors were not able to load");
      continue;
    }

    // The input to the newly created MatMul or MatMulNBits node.
    NodeArg* layer_norm = q_node->MutableInputDefs()[0];

    const onnx::TypeProto* layer_norm_tensor_proto = layer_norm->TypeAsProto();
    onnx::TypeProto mutable_matmul_or_nbits_tensor_proto = *layer_norm_tensor_proto;
    auto* matmul_or_nbits_tensor_type = mutable_matmul_or_nbits_tensor_proto.mutable_tensor_type();
    auto* matmul_or_nbits_output_shape = matmul_or_nbits_tensor_type->mutable_shape();

    int64_t head_size = past_key_values_key_arg->Shape()->dim(3).dim_value();
    int64_t num_heads = node.GetAttributes().at("num_heads").i();
    int64_t kv_num_heads = node.GetAttributes().at("kv_num_heads").i();
    int64_t q_hidden_size = num_heads * head_size;
    int64_t kv_hidden_size = kv_num_heads * head_size;
    int64_t output_hidden_size = q_hidden_size + 2 * kv_hidden_size;

    // Ensure the output shape has 3 dimensions [batch_size, sequence_length, hidden_size]
    if (matmul_or_nbits_output_shape->dim_size() == 3) {
      auto* third_dim = matmul_or_nbits_output_shape->mutable_dim(2);
      third_dim->set_dim_value(output_hidden_size);
    } else {
      DEBUG_LOG("The newly created node does not follow output def shape of [batch_size, sequence_length, hidden_size]");
      continue;
    }

    auto& matmul_or_nbits_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MatMul_output"), &mutable_matmul_or_nbits_tensor_proto);
    const std::array mmnb_output_defs{&matmul_or_nbits_output};

    Node* mat_mul_or_n_bits_new_node = nullptr;

    if (!quantization_used) {
      auto& qkv_weights = MergeQkvWeightsForMatMul(graph, q_hidden_size, kv_hidden_size, q_proj_tensor, k_proj_tensor, v_proj_tensor);
      std::array mmnb_input_defs{layer_norm, &qkv_weights};

      mat_mul_or_n_bits_new_node = &graph.AddNode(graph.GenerateNodeName("MatMul"),
                                                  "MatMul",
                                                  "MatMul fusion node",
                                                  mmnb_input_defs,
                                                  mmnb_output_defs,
                                                  &q_node->GetAttributes(),
                                                  kOnnxDomainAlias);
    } else {
      auto qkv_args = MergeQkvWeightsForMatMulNBits(
          graph,
          q_hidden_size,
          kv_hidden_size,
          q_proj_tensor->dims(1),
          q_proj_tensor->dims(2),
          q_proj_tensor,
          k_proj_tensor,
          v_proj_tensor,
          q_scale_tensor,
          q_zero_points_tensor,
          k_scale_tensor,
          k_zero_points_tensor,
          v_scale_tensor,
          v_zero_points_tensor);

      // If the zero points tensor was present.
      if (qkv_args.size() == 3) {
        const std::array mmnb_input_defs = {layer_norm, qkv_args[0], qkv_args[1], qkv_args[2]};

        mat_mul_or_n_bits_new_node = &graph.AddNode(graph.GenerateNodeName("MatMulNBits"),
                                                    "MatMulNBits",
                                                    "MatMulNBits fusion node",
                                                    mmnb_input_defs,
                                                    mmnb_output_defs,
                                                    &q_node->GetAttributes(),
                                                    kMSDomain);

      } else {
        const std::array mmnb_input_defs = {layer_norm, qkv_args[0], qkv_args[1]};

        mat_mul_or_n_bits_new_node = &graph.AddNode(graph.GenerateNodeName("MatMulNBits"),
                                                    "MatMulNBits",
                                                    "MatMulNBits fusion node",
                                                    mmnb_input_defs,
                                                    mmnb_output_defs,
                                                    &q_node->GetAttributes(),
                                                    kMSDomain);
      }

      mat_mul_or_n_bits_new_node->GetMutableAttributes()["N"] = ONNX_NAMESPACE::MakeAttribute("N", static_cast<int64_t>(output_hidden_size));
    }

    mat_mul_or_n_bits_new_node->SetExecutionProviderType(node.GetExecutionProviderType());
    FusePreGQANodes(graph, q_node, k_node, v_node, rotary_node_1, rotary_node_2, mat_mul_or_n_bits_new_node, matmul_or_nbits_output);

    node.GetMutableAttributes()["do_rotary"] = ONNX_NAMESPACE::MakeAttribute("do_rotary", static_cast<int64_t>(1));

    std::string empty_name;
    auto& empty_node_arg = graph.GetOrCreateNodeArg(empty_name, nullptr);

    const std::array gqa_input_defs{
        &matmul_or_nbits_output,
        &empty_node_arg,
        &empty_node_arg,
        past_key_values_key_arg,
        past_key_values_value_arg,
        seqlens_k,
        total_seq_len,
        cos_cache_arg,
        sin_cache_arg};

    auto& gqa_input_args = node.MutableInputArgsCount();
    gqa_input_args[7] = 1;
    gqa_input_args[8] = 1;

    // Switch GQA input defs from unfused into the fused form.
    auto& gqa_node_input_defs = node.MutableInputDefs();
    gqa_node_input_defs.assign(gqa_input_defs.begin(), gqa_input_defs.end());

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
