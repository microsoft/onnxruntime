// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/optimizer/megatron_transformer.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/random_seed.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

struct OpInfo {
  OpInfo(const std::string& op_type,
         const std::initializer_list<OperatorSetVersion>& supported_versions,
         const std::string& domain = kOnnxDomainAlias,
         const size_t output_count = 1) : op_type(op_type),
                                          supported_versions(supported_versions),
                                          domain(domain),
                                          output_count(output_count){};

  std::string op_type;
  std::initializer_list<OperatorSetVersion> supported_versions;
  std::string domain;
  size_t output_count;
};

const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v1 = {1};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v1_11 = {1, 11};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v2_11 = {2, 11};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v5 = {5};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v7 = {7};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v9 = {9};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v12 = {12};
const OpInfo add_info = OpInfo("Add", opset_v7);
const OpInfo split_info = OpInfo("Split", opset_v2_11, kOnnxDomainAlias, 3);
const OpInfo reshape_info = OpInfo("Reshape", opset_v5);
const OpInfo transpose_info = OpInfo("Transpose", opset_v1);
const OpInfo matmul_info = OpInfo("MatMul", opset_v9);
const OpInfo div_info = OpInfo("Div", opset_v7);
const OpInfo mul_info = OpInfo("Mul", opset_v7);
const OpInfo sub_info = OpInfo("Sub", opset_v7);
const OpInfo softmax_info = OpInfo("Softmax", opset_v1_11);
const OpInfo trainable_dropout_info = OpInfo("TrainableDropout", opset_v9, kOnnxDomain);
const OpInfo dropout_info = OpInfo("Dropout", opset_v12);

struct NodeInfo {
  NodeInfo(const std::vector<OpInfo>& op_infos,
           const bool required = true) : op_infos(op_infos),
                                         required(required){};

  std::vector<OpInfo> op_infos;
  bool required;
};

// Check if it's an expected node given the op infos and provider type.
static bool IsExpectedOpAndProvider(const Node& node,
                                    const OpInfo& op_info,
                                    ProviderType provider_type) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, op_info.op_type, op_info.supported_versions, op_info.domain) &&
         node.GetExecutionProviderType() == provider_type &&
         node.GetOutputEdgesCount() == op_info.output_count;
}

// Try to match a linear sub-graph pattern given a list of graph node infos, input node excluded.
// Each node info entry contains a vector of all possible Op infos, and a flag of required or not.
// All visited nodes will be pushed back to a given node vector.
static bool MatchLinearPattern(Graph& graph,
                               Node* node,
                               ProviderType provider_type,
                               const std::vector<NodeInfo>& node_infos,
                               std::vector<Node*>& sub_graph_node_ptrs) {
  Node* curr_node_ptr = node;
  if (curr_node_ptr->GetOutputEdgesCount() == 0) {
    return node_infos.size() == 0;
  }

  for (const auto& node_info : node_infos) {
    Node* next_node_ptr = graph.GetNode(curr_node_ptr->OutputNodesBegin()->Index());
    bool has_matched_op = false;
    for (const auto& op_info : node_info.op_infos) {
      if (IsExpectedOpAndProvider(*next_node_ptr, op_info, provider_type)) {
        has_matched_op = true;
        break;
      }
    }

    sub_graph_node_ptrs.push_back(has_matched_op ? next_node_ptr : nullptr);
    if (has_matched_op) {
      curr_node_ptr = next_node_ptr;
    } else if (node_info.required) {
      return false;
    }
  }

  return true;
}

// std::hash only guarantee deterministic value in single execution of a program.
// So use this simple hash to generate dropout seed by name.
static uint32_t HashName(const std::string& name) {
  uint32_t hash = 0;
  for (char const& c : name) {
    hash = hash * 101 + c;
  }

  return hash;
}

bool MegatronTransformer::PartitionWeightByColumn(const Graph& graph, const NodeArg& input_arg,
                                                  ONNX_NAMESPACE::TensorProto& initializer_partition,
                                                  int stride) const {
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  if (!graph.GetInitializedTensor(input_arg.Name(), tensor_proto)) {
    LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn: " << input_arg.Name() << " is not an initializer";
    return false;
  }

  auto data_type = tensor_proto->data_type();
  const ONNX_NAMESPACE::TensorShapeProto* shape = input_arg.Shape();
  int rank = shape->dim_size();
  int64_t row_count;
  int64_t column_count;

  if (rank == 2 && utils::HasDimValue(shape->dim(0)) && utils::HasDimValue(shape->dim(1))) {
    row_count = shape->dim(0).dim_value();
    column_count = shape->dim(1).dim_value();
  } else if (rank == 1) {
    row_count = 1;
    column_count = shape->dim(0).dim_value();
  } else {
    LOGS_DEFAULT(WARNING) << "Initializer tensor's rank is " << rank << " (expected to be 1 or 2).";
    return false;
  }

  if (column_count % (horizontal_parallel_size_ * stride) != 0) {
    LOGS_DEFAULT(WARNING) << "last dim " << column_count
                          << " is not divisible by horizontal_parallel_size_ times stride "
                          << (horizontal_parallel_size_ * stride) << ", not supported currently.";
    return false;
  }

  auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
  const float* a_weight = initializer->data<float>();

  initializer_partition.set_name("rank_" + std::to_string(horizontal_parallel_rank_) +
                                 "_" + input_arg.Name() + "_partition");
  initializer_partition.set_data_type(data_type);

  int64_t column_partition = column_count / horizontal_parallel_size_;
  int64_t column_stride = column_count / stride;
  int64_t column_stride_partition = column_stride / horizontal_parallel_size_;

  if (rank == 2) {
    initializer_partition.add_dims(row_count);
  }

  initializer_partition.add_dims(column_partition);
  const int64_t element_count = row_count * column_partition;

  std::vector<float> result;
  result.reserve(element_count);

  const int64_t stride_partition_column_offset = horizontal_parallel_rank_ * column_stride_partition;
  for (auto row_index = 0; row_index < row_count; row_index++) {
    auto row_offset = row_index * column_count;
    for (auto stride_index = 0; stride_index < stride; stride_index++) {
      auto column_offset = row_offset + stride_index * column_stride + stride_partition_column_offset;
      std::copy(a_weight + column_offset, a_weight + column_offset + column_stride_partition, std::back_inserter(result));
    }
  }

  initializer_partition.set_raw_data(result.data(), element_count * sizeof(float));
  return true;
}

bool MegatronTransformer::PartitionWeightByRow(const Graph& graph, const NodeArg& input_arg,
                                               ONNX_NAMESPACE::TensorProto& initializer_partition) const {
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  if (!graph.GetInitializedTensor(input_arg.Name(), tensor_proto)) {
    LOGS_DEFAULT(WARNING) << "PartitionWeightByRow: " << input_arg.Name() << " is not an initializer";
    return false;
  }

  auto data_type = tensor_proto->data_type();
  const ONNX_NAMESPACE::TensorShapeProto* shape = input_arg.Shape();
  int rank = shape->dim_size();
  int64_t row_count;
  int64_t column_count;

  if (rank == 2 && utils::HasDimValue(shape->dim(0)) && utils::HasDimValue(shape->dim(1))) {
    row_count = shape->dim(0).dim_value();
    column_count = shape->dim(1).dim_value();
  } else if (rank == 1) {
    row_count = shape->dim(0).dim_value();
    column_count = 1;
  } else {
    LOGS_DEFAULT(WARNING) << "Initializer tensor's rank is more than " << rank
                          << " (expected to be 1 or 2).";
    return false;
  }

  if (row_count % horizontal_parallel_size_ != 0) {
    LOGS_DEFAULT(WARNING) << "first dim " << row_count << " is not divisible by horizontal parallel size"
                          << horizontal_parallel_size_ << ", not supported currently.";
    return false;
  }

  auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
  const float* a_weight = initializer->data<float>();

  initializer_partition.set_name("rank_" + std::to_string(horizontal_parallel_rank_) +
                                 "_" + input_arg.Name() + "_partition");
  initializer_partition.set_data_type(data_type);

  int64_t row_partition = row_count / horizontal_parallel_size_;

  initializer_partition.add_dims(row_partition);
  if (rank == 2) {
    initializer_partition.add_dims(column_count);
  }
  const int64_t element_count = row_partition * column_count;

  std::vector<float> result;
  result.reserve(element_count);

  const int64_t row_index_offset = horizontal_parallel_rank_ * row_partition;
  memcpy(result.data(), a_weight + row_index_offset * column_count, sizeof(float) * element_count);
  initializer_partition.set_raw_data(result.data(), element_count * sizeof(float));

  return true;
}

Status MegatronTransformer::TransformMLP(Graph& graph, bool& modified, int graph_level,
                                         const logging::Logger& logger,
                                         std::vector<Node*>& nodes_to_clear_shape) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {9}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    if (node.GetInputEdgesCount() > 0) {
      Node& matmul_input_node = const_cast<Node&>(*(node.InputNodesBegin()));
      if (matmul_input_node.OpType().compare("MegatronF") == 0) {
        continue;
      }
    }

    Node& add_node = *graph.GetNode(node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7}) ||
        add_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        add_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    Node& gelu_node = *graph.GetNode(add_node.OutputNodesBegin()->Index());
    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(gelu_node, "Gelu", {1}, kMSDomain) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(gelu_node, "FastGelu", {1}, kMSDomain)) ||
        gelu_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        gelu_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    Node& matmul2_node = *graph.GetNode(gelu_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(matmul2_node, "MatMul", {9}) ||
        matmul2_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        matmul2_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    Node& add2_node = *graph.GetNode(matmul2_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add2_node, "Add", {7}) ||
        add2_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        add2_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    nodes_to_clear_shape.insert(nodes_to_clear_shape.end(), {&node, &add_node, &gelu_node,
                                                             &matmul2_node});

    auto a_weight_arg = node.MutableInputDefs()[1];
    ONNX_NAMESPACE::TensorProto a_weight_initializer_partition;
    if (!PartitionWeightByColumn(graph, *a_weight_arg, a_weight_initializer_partition)) {
      continue;
    }

    auto a_bias_arg = add_node.MutableInputDefs()[1];
    ONNX_NAMESPACE::TensorProto a_bias_initializer_partition;
    if (!PartitionWeightByColumn(graph, *a_bias_arg, a_bias_initializer_partition)) {
      continue;
    }

    auto b_weight_arg = matmul2_node.MutableInputDefs()[1];
    ONNX_NAMESPACE::TensorProto b_weight_initializer_partition;
    if (!PartitionWeightByRow(graph, *b_weight_arg, b_weight_initializer_partition)) {
      continue;
    }

    NodeArg& a_weight_partition_arg = graph_utils::AddInitializer(graph, a_weight_initializer_partition);
    graph_utils::ReplaceNodeInput(node, 1, a_weight_partition_arg);

    NodeArg& a_bias_partition_arg = graph_utils::AddInitializer(graph, a_bias_initializer_partition);
    graph_utils::ReplaceNodeInput(add_node, 1, a_bias_partition_arg);

    NodeArg& b_weight_partition_arg = graph_utils::AddInitializer(graph, b_weight_initializer_partition);
    graph_utils::ReplaceNodeInput(matmul2_node, 1, b_weight_partition_arg);

    graph.RemoveInitializedTensor(a_weight_arg->Name());
    graph.RemoveInitializedTensor(b_weight_arg->Name());
    graph.RemoveInitializedTensor(a_bias_arg->Name());

    const std::vector<NodeArg*> mlp_f_input_defs{node.MutableInputDefs()[0]};
    auto mlp_f_type_info = *node.MutableInputDefs()[0]->TypeAsProto();
    auto& mlp_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MLP_MegatronF_Output"), &mlp_f_type_info);
    Node& mlp_f_node = graph.AddNode(graph.GenerateNodeName("MLP_MegatronF"),
                                     "MegatronF",
                                     "MLP MegatronF",
                                     mlp_f_input_defs,
                                     {&mlp_f_out_arg}, {}, kMSDomain);
    mlp_f_node.SetExecutionProviderType(node.GetExecutionProviderType());
    const Node::EdgeEnd* edge = graph_utils::GetInputEdge(node, 0);
    if (nullptr == edge) {  // handle input/initializer
      graph_utils::ReplaceNodeInput(node, 0, *(mlp_f_node.MutableOutputDefs()[0]));
    } else {
      auto input_node = const_cast<Node*>(&edge->GetNode());
      graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge->GetDstArgIndex(), mlp_f_node, 0);
    }

    const std::vector<NodeArg*> mlp_g_input_defs{matmul2_node.MutableOutputDefs()[0]};
    auto mlp_g_type_info = *matmul2_node.MutableOutputDefs()[0]->TypeAsProto();
    auto& mlp_g_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MLP_MegatronG_Output"), &mlp_g_type_info);
    Node& mlp_g_node = graph.AddNode(graph.GenerateNodeName("MLP_MegatronG"),
                                     "MegatronG",
                                     "MLP MegatronG",
                                     mlp_g_input_defs,
                                     {&mlp_g_out_arg}, {}, kMSDomain);
    mlp_g_node.AddAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel));
    mlp_g_node.SetExecutionProviderType(node.GetExecutionProviderType());
    graph_utils::ReplaceDownstreamNodeInput(graph, matmul2_node, 0, mlp_g_node, 0);
    modified = true;
  }

  return Status::OK();
}

Status MegatronTransformer::TransformSelfAttention(Graph& graph, bool& modified, int graph_level,
                                                   const logging::Logger& logger,
                                                   std::vector<Node*>& nodes_to_clear_shape,
                                                   std::unordered_set<Node*>& self_attention_dropout_nodes) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // Self attention sub-graph.
  // MatMul->Add->Split->Reshape->Transpose->MatMul->Div->Mul->Sub->Softmax->Dropout->MatMul->Transpose->Reshape->MatMul->Add
  //                  |->Reshape->Transpose->|                                        |
  //                  |->Reshape->Transpose------------------------------------------>|
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", opset_v9) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    if (node.GetInputEdgesCount() > 0 && node.InputNodesBegin()->OpType().compare("MegatronF") == 0) {
      continue;
    }

    std::vector<Node*> sub_graph_node_ptrs;
    sub_graph_node_ptrs.push_back(&node);
    ProviderType provider_type = node.GetExecutionProviderType();

    std::vector<NodeInfo> linear_pattern = {
        NodeInfo({add_info}),  // -15
        NodeInfo({split_info}),
        NodeInfo({reshape_info}),
        NodeInfo({transpose_info}),
        NodeInfo({matmul_info}),  // -11
        NodeInfo({div_info}),
        NodeInfo({mul_info}),
        NodeInfo({sub_info}),
        NodeInfo({softmax_info}),
        NodeInfo({trainable_dropout_info, dropout_info}, false),  // -6
        NodeInfo({matmul_info}),
        NodeInfo({transpose_info}),
        NodeInfo({reshape_info}),
        NodeInfo({matmul_info}),
        NodeInfo({add_info})};  // -1
    if (!MatchLinearPattern(graph, &node, provider_type, linear_pattern, sub_graph_node_ptrs)) {
      continue;
    }

    // Get all useful nodes here as more vector push back below will change the index.
    Node& add_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 15];
    Node& split_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 14];
    Node& transpose_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 12];
    Node* matmul_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 11];
    Node* dropout_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 6];
    Node* matmul_node_ptr1 = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 5];
    Node& transpose_node1 = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 4];
    Node& matmul_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 2];

    // Transpose node attribute checking.
    if (!optimizer_utils::IsAttributeWithExpectedValues(transpose_node, "perm", {0LL, 2LL, 1LL, 3LL}) ||
        !optimizer_utils::IsAttributeWithExpectedValues(transpose_node1, "perm", {0LL, 2LL, 1LL, 3LL})) {
      continue;
    }

    std::vector<Node*> transpose_node_ptrs;  // For the 2nd and 3rd transpose nodes after split node for sub-graph structure checking.
    std::vector<Node*> reshape_node_ptrs;    // To keep the reshape node that need to change the shape constant.
    reshape_node_ptrs.push_back(sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 13]);
    reshape_node_ptrs.push_back(sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 3]);
    auto split_output_iter = split_node.OutputNodesBegin();
    ++split_output_iter;
    for (; split_output_iter != split_node.OutputNodesEnd(); ++split_output_iter) {
      Node* reshape_node_ptr = graph.GetNode((*split_output_iter).Index());
      if (!IsExpectedOpAndProvider(*reshape_node_ptr, reshape_info, provider_type)) {
        break;
      }

      Node* transpose_node_ptr = graph.GetNode(reshape_node_ptr->OutputNodesBegin()->Index());
      if (!IsExpectedOpAndProvider(*transpose_node_ptr, transpose_info, provider_type)) {
        break;
      }

      reshape_node_ptrs.push_back(reshape_node_ptr);
      sub_graph_node_ptrs.push_back(reshape_node_ptr);
      transpose_node_ptrs.push_back(transpose_node_ptr);
      sub_graph_node_ptrs.push_back(transpose_node_ptr);
    }

    // Sub-graph structure and transpose attribute checking.
    if (transpose_node_ptrs.size() != 2 ||
        matmul_node_ptr != graph.GetNode(transpose_node_ptrs[0]->OutputNodesBegin()->Index()) ||
        matmul_node_ptr1 != graph.GetNode(transpose_node_ptrs[1]->OutputNodesBegin()->Index()) ||
        !optimizer_utils::IsAttributeWithExpectedValues(*transpose_node_ptrs[0], "perm", {0LL, 2LL, 3LL, 1LL}) ||
        !optimizer_utils::IsAttributeWithExpectedValues(*transpose_node_ptrs[1], "perm", {0LL, 2LL, 1LL, 3LL})) {
      continue;
    }

    // Partition weights. If any of them fails, skip transforming this sub-graph.
    auto qkv_weight_arg = node.MutableInputDefs()[1];
    ONNX_NAMESPACE::TensorProto qkv_weight_initializer_partition;
    if (!PartitionWeightByColumn(graph, *qkv_weight_arg, qkv_weight_initializer_partition, 3)) {
      continue;
    }

    auto qkv_bias_arg = add_node.MutableInputDefs()[1];
    ONNX_NAMESPACE::TensorProto qkv_bias_initializer_partition;
    if (!PartitionWeightByColumn(graph, *qkv_bias_arg, qkv_bias_initializer_partition, 3)) {
      continue;
    }

    auto dense_weight_arg = matmul_node.MutableInputDefs()[1];
    ONNX_NAMESPACE::TensorProto dense_weight_initializer_partition;
    if (!PartitionWeightByRow(graph, *dense_weight_arg, dense_weight_initializer_partition)) {
      continue;
    }

    // Check the constant value in the Reshape nodes.
    bool is_reshape_valid = true;
    for (Node* node_ptr : reshape_node_ptrs) {
      auto shape_arg = node_ptr->MutableInputDefs()[1];
      const ONNX_NAMESPACE::TensorProto* tensor;

      if (!graph.GetInitializedTensor(shape_arg->Name(), tensor)) {
        is_reshape_valid = false;
        break;
      }

      auto data_type = tensor->data_type();
      if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        is_reshape_valid = false;
        break;
      }

      // The number of the values should be more than 2, and the 3rd value should be divisible by parallel size,
      // i.e., the attention head number should be divisible by parallel size.
      auto init_const = onnxruntime::make_unique<Initializer>(*tensor, graph.ModelPath());
      if (init_const->size() != 3 && init_const->size() != 4) {
        is_reshape_valid = false;
        break;
      }

      const int64_t* val = init_const->data<int64_t>();
      if (val[2] % horizontal_parallel_size_ != 0) {
        LOGS_DEFAULT(WARNING) << (init_const->size() == 3 ? "Hidden size " : "Number of attention heads ") << val[2]
                              << " is not divisible by horizontal_parallel_size_ "
                              << horizontal_parallel_size_ << ", not supported currently.";
        is_reshape_valid = false;
        break;
      }
    }

    if (!is_reshape_valid) {
      continue;
    }

    // Ready to transform the sub-graph when reach here.
    // It's possible that the node vector contains nullptr due to some optinal node infos during linear pattern matching.
    std::copy_if(sub_graph_node_ptrs.begin(), sub_graph_node_ptrs.end(),
                 std::back_inserter(nodes_to_clear_shape),
                 [](Node* node_ptr) { return node_ptr != nullptr; });

    // Replace by the partition weights.
    NodeArg& qkv_weight_partition_arg = graph_utils::AddInitializer(graph, qkv_weight_initializer_partition);
    graph_utils::ReplaceNodeInput(node, 1, qkv_weight_partition_arg);

    NodeArg& qkv_bias_partition_arg = graph_utils::AddInitializer(graph, qkv_bias_initializer_partition);
    graph_utils::ReplaceNodeInput(add_node, 1, qkv_bias_partition_arg);

    NodeArg& dense_weight_partition_arg = graph_utils::AddInitializer(graph, dense_weight_initializer_partition);
    graph_utils::ReplaceNodeInput(matmul_node, 1, dense_weight_partition_arg);

    graph.RemoveInitializedTensor(qkv_weight_arg->Name());
    graph.RemoveInitializedTensor(qkv_bias_arg->Name());
    graph.RemoveInitializedTensor(dense_weight_arg->Name());

    // Change the constant for the reshape nodes.
    for (Node* node_ptr : reshape_node_ptrs) {
      auto shape_arg = node_ptr->MutableInputDefs()[1];
      const ONNX_NAMESPACE::TensorProto* tensor;
      graph.GetInitializedTensor(shape_arg->Name(), tensor);
      auto data_type = tensor->data_type();
      auto init_const = onnxruntime::make_unique<Initializer>(*tensor, graph.ModelPath());
      const int64_t* val = init_const->data<int64_t>();
      int64_t size = init_const->size();
      ONNX_NAMESPACE::TensorProto tensor_partition;
      tensor_partition.set_name(graph.GenerateNodeArgName("partition_" + shape_arg->Name()));
      tensor_partition.set_data_type(data_type);
      tensor_partition.add_dims(size);

      std::vector<int64_t> val_partition;
      val_partition.reserve(size);
      val_partition.insert(val_partition.end(), val, val + size);
      val_partition[2] /= horizontal_parallel_size_;
      tensor_partition.set_raw_data(val_partition.data(), size * sizeof(int64_t));
      NodeArg& node_arg_partition = graph_utils::AddInitializer(graph, tensor_partition);
      graph_utils::ReplaceNodeInput(*node_ptr, 1, node_arg_partition);
      graph.RemoveInitializedTensor(shape_arg->Name());
    }

    if (dropout_node_ptr != nullptr) {
      self_attention_dropout_nodes.insert(dropout_node_ptr);
    }

    // Add MegatronF before the 1st MatMul and MegatronG before the last Add.
    const std::vector<NodeArg*> sa_f_input_defs{node.MutableInputDefs()[0]};
    auto sa_f_type_info = *node.MutableInputDefs()[0]->TypeAsProto();
    auto& sa_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("SelfAttention_MegatronF_Output"), &sa_f_type_info);
    Node& sa_f_node = graph.AddNode(graph.GenerateNodeName("SelfAttention_MegatronF"),
                                    "MegatronF",
                                    "SelfAttention MegatronF",
                                    sa_f_input_defs,
                                    {&sa_f_out_arg}, {}, kMSDomain);
    sa_f_node.SetExecutionProviderType(node.GetExecutionProviderType());
    const Node::EdgeEnd* edge = graph_utils::GetInputEdge(node, 0);
    if (nullptr == edge) {  // handle input/initializer
      graph_utils::ReplaceNodeInput(node, 0, *(sa_f_node.MutableOutputDefs()[0]));
    } else {
      auto input_node = const_cast<Node*>(&edge->GetNode());
      graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge->GetDstArgIndex(), sa_f_node, 0);
    }

    const std::vector<NodeArg*> sa_g_input_defs{matmul_node.MutableOutputDefs()[0]};
    auto sa_g_type_info = *matmul_node.MutableOutputDefs()[0]->TypeAsProto();  // copy
    auto& sa_g_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("SelfAttention_MegatronG_Output"), &sa_g_type_info);
    Node& sa_g_node = graph.AddNode(graph.GenerateNodeName("SelfAttention_MegatronG"),
                                    "MegatronG",
                                    "SelfAttention MegatronG",
                                    sa_g_input_defs,
                                    {&sa_g_out_arg}, {}, kMSDomain);
    sa_g_node.AddAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel));
    sa_g_node.SetExecutionProviderType(node.GetExecutionProviderType());
    graph_utils::ReplaceDownstreamNodeInput(graph, matmul_node, 0, sa_g_node, 0);
    modified = true;
  }

  return Status::OK();
}

Status MegatronTransformer::TransformDropout(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger,
                                             std::unordered_set<Node*>& self_attention_dropout_nodes) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Dropout", opset_v12) &&
        !graph_utils::IsSupportedOptypeVersionAndDomain(node, "TrainableDropout", opset_v9, kOnnxDomain)) {
      continue;
    }

    // Only need to set the seed if it's a transformed self-attention dropout, or the seed attribute is not set.
    if (self_attention_dropout_nodes.find(&node) != self_attention_dropout_nodes.end() ||
        graph_utils::GetNodeAttribute(node, "seed") == nullptr) {
      int64_t seed = static_cast<int64_t>(HashName(node.MutableOutputDefs()[0]->Name())) + utils::GetRandomSeed();
      if (self_attention_dropout_nodes.find(&node) != self_attention_dropout_nodes.end()) {
        seed += horizontal_parallel_rank_;
      }

      if (graph_utils::GetNodeAttribute(node, "seed") != nullptr) {
        node.ClearAttribute("seed");
      }

      node.AddAttribute("seed", seed);
      modified = true;
    }
  }

  return Status::OK();
}

Status MegatronTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  if (horizontal_parallel_size_ <= 1) {
    return Status::OK();
  }

  std::vector<Node*> nodes_to_clear_shape;
  std::unordered_set<Node*> self_attention_dropout_nodes;

  ORT_ENFORCE(TransformMLP(graph, modified, graph_level, logger, nodes_to_clear_shape).IsOK());
  ORT_ENFORCE(TransformSelfAttention(graph, modified, graph_level, logger, nodes_to_clear_shape, self_attention_dropout_nodes).IsOK());
  ORT_ENFORCE(TransformDropout(graph, modified, graph_level, logger, self_attention_dropout_nodes).IsOK());

  auto& graph_inputs = graph.GetInputs();
  for (auto& node : nodes_to_clear_shape) {
    auto& inputs = node->MutableInputDefs();
    for (auto* input : inputs)
      if (std::find(graph_inputs.begin(), graph_inputs.end(), input) == graph_inputs.end())
        input->ClearShape();

    for (auto* output : node->MutableOutputDefs())
      if (std::find(graph_inputs.begin(), graph_inputs.end(), output) == graph_inputs.end())
        output->ClearShape();
  }

  if (modified) {
    graph.SetGraphResolveNeeded();
    auto ret = graph.Resolve();
    ORT_ENFORCE(ret.IsOK());
  }

  return Status::OK();
}

}  // namespace onnxruntime
