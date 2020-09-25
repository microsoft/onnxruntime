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
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v1_6_7 = {1, 6, 7};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v7 = {7};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v9 = {9};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v12 = {12};
const OpInfo add_info = OpInfo("Add", opset_v7);
const OpInfo split_info = OpInfo("Split", opset_v2_11, kOnnxDomainAlias, 3);
const OpInfo reshape_info = OpInfo("Reshape", opset_v5);
const OpInfo transpose_info = OpInfo("Transpose", opset_v1);
const OpInfo matmul_info = OpInfo("MatMul", opset_v9);
const OpInfo div_info = OpInfo("Div", opset_v7);
const OpInfo mul_info = OpInfo("Mul", opset_v1_6_7);
const OpInfo sub_info = OpInfo("Sub", opset_v7);
const OpInfo softmax_info = OpInfo("Softmax", opset_v1_11);
const OpInfo trainable_dropout_info = OpInfo("TrainableDropout", opset_v9, kOnnxDomain);
const OpInfo dropout_info = OpInfo("Dropout", opset_v12);
const OpInfo where_info = OpInfo("Where", opset_v9);

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
  if (node.OpType() == "Mul") {
    std::cout << "Extra debug:" << node.Name() << "\n";
    bool is_true = graph_utils::IsSupportedOptypeVersionAndDomain(node, op_info.op_type, op_info.supported_versions, op_info.domain);
    if (!is_true) {
      std::cout << "Mismatch in IsSupportedOptypeVersionAndDomain:\n";
      return is_true;
    }
    is_true = node.GetExecutionProviderType() == provider_type;
    if (!is_true) {
      std::cout << "Mismatch in ExecutionProviderType:\n";
      return is_true;
    }
    is_true = node.GetOutputEdgesCount() == op_info.output_count;
    if (!is_true) {
      std::cout << "Mismatch in IsSupportedOptypeVersionAndDomain:\n";
      return is_true;
    }
    return true;
  }
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
      LOGS_DEFAULT(WARNING) << "  MatchLinearPattern 2222" << next_node_ptr->Name();
      curr_node_ptr = next_node_ptr;
    } else if (node_info.required) {
      LOGS_DEFAULT(WARNING) << "  MatchLinearPattern 11111" << next_node_ptr->Name();
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

NodeArg& MegatronTransformer::PartitionWeightByColumn(Graph& graph, const NodeArg& input_arg,
                                                      int stride) const {
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn 1111111111111111111" << input_arg.Name();
  ONNX_NAMESPACE::TensorProto initializer_partition;
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  ORT_ENFORCE(graph.GetInitializedTensor(input_arg.Name(), tensor_proto),
              "PartitionWeightByColumn: " + input_arg.Name() + " is not an initializer");
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn 2222222222222222222" << input_arg.Name();
  auto data_type = tensor_proto->data_type();
  const ONNX_NAMESPACE::TensorShapeProto* shape = input_arg.Shape();
  int rank = shape->dim_size();
  int64_t row_count;
  int64_t column_count;
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn 33333333333333" << input_arg.Name();
  if (rank == 2 && utils::HasDimValue(shape->dim(0)) && utils::HasDimValue(shape->dim(1))) {
    row_count = shape->dim(0).dim_value();
    column_count = shape->dim(1).dim_value();
  } else if (rank == 1) {
    row_count = 1;
    column_count = shape->dim(0).dim_value();
  } else {
    ORT_THROW("Initializer tensor's rank is " + std::to_string(rank) + " (expected to be 1 or 2).");
  }
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn 4444444444444444444" << input_arg.Name();
  ORT_ENFORCE(column_count % (horizontal_parallel_size_ * stride) == 0,
              "last dim " + std::to_string(column_count) +
                  " is not divisible by horizontal_parallel_size_ times stride " +
                  std::to_string(horizontal_parallel_size_ * stride) + ", not supported currently.");

  auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
  const float* a_weight = initializer->data<float>();

  std::string new_initializer_name = input_arg.Name() + "_column_rank_" + std::to_string(horizontal_parallel_rank_);
  updated_weight_names_.insert({input_arg.Name(), new_initializer_name});
  if (weights_to_train_.find(input_arg.Name()) != weights_to_train_.end()) {
    weights_to_train_.erase(input_arg.Name());
    weights_to_train_.insert(new_initializer_name);
  }
  initializer_partition.set_name(new_initializer_name);
  initializer_partition.set_data_type(data_type);

  int64_t column_partition = column_count / horizontal_parallel_size_;
  int64_t column_stride = column_count / stride;
  int64_t column_stride_partition = column_stride / horizontal_parallel_size_;
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn 555555555555555555555555555" << input_arg.Name();
  if (rank == 2) {
    initializer_partition.add_dims(row_count);
  }

  initializer_partition.add_dims(column_partition);
  const int64_t element_count = row_count * column_partition;
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn 6666666666666666666666" << input_arg.Name();
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
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn 777777777777777777777" << input_arg.Name();
  initializer_partition.set_raw_data(result.data(), element_count * sizeof(float));
  graph.RemoveInitializedTensor(input_arg.Name());
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn 888888888888888888888888" << input_arg.Name();
  NodeArg& partition_arg = graph_utils::AddInitializer(graph, initializer_partition);
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn 999999999999999" << input_arg.Name();
  // need clear shape, because the node arge is reusing previous node arg due to same name.
  partition_arg.ClearShape();
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn " << input_arg.Name();
  return partition_arg;
}

NodeArg& MegatronTransformer::PartitionWeightByRow(Graph& graph, const NodeArg& input_arg) const {
  ONNX_NAMESPACE::TensorProto initializer_partition;
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  ORT_ENFORCE(graph.GetInitializedTensor(input_arg.Name(), tensor_proto),
              "PartitionWeightByRow: " + input_arg.Name() + " is not an initializer");

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
    ORT_THROW("Initializer tensor's rank is more than " + std::to_string(rank) +
              " (expected to be 1 or 2).");
  }

  ORT_ENFORCE(row_count % horizontal_parallel_size_ == 0, "first dim " + std::to_string(row_count) +
                                                              " is not divisible by horizontal parallel size" +
                                                              std::to_string(horizontal_parallel_size_) +
                                                              ", not supported currently.");
  auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
  const float* a_weight = initializer->data<float>();

  std::string new_initializer_name = input_arg.Name() + "_row_rank_" + std::to_string(horizontal_parallel_rank_);
  updated_weight_names_.insert({input_arg.Name(), new_initializer_name});
  if (weights_to_train_.find(input_arg.Name()) != weights_to_train_.end()) {
    weights_to_train_.erase(input_arg.Name());
    weights_to_train_.insert(new_initializer_name);
  }
  initializer_partition.set_name(new_initializer_name);
  initializer_partition.set_data_type(data_type);

  int64_t row_partition = row_count / horizontal_parallel_size_;

  initializer_partition.add_dims(row_partition);
  if (rank == 2) {
    initializer_partition.add_dims(column_count);
  }

  //std::cout << input_arg.Name() << "  " << row_partition << "x" << column_count << std::endl;
  const int64_t element_count = row_partition * column_count;

  std::vector<float> result;
  result.reserve(element_count);

  const int64_t row_index_offset = horizontal_parallel_rank_ * row_partition;
  memcpy(result.data(), a_weight + row_index_offset * column_count, sizeof(float) * element_count);
  initializer_partition.set_raw_data(result.data(), element_count * sizeof(float));

  graph.RemoveInitializedTensor(input_arg.Name());

  NodeArg& partition_arg = graph_utils::AddInitializer(graph, initializer_partition);
  // need clear shape, because the node arge is reusing previous node arg due to same name.
  partition_arg.ClearShape();
  //LOGS_DEFAULT(WARNING) << "PartitionWeightByRow " << input_arg.Name();
  return partition_arg;
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
    NodeArg& a_weight_partition_arg = PartitionWeightByColumn(graph, *a_weight_arg);

    auto a_bias_arg = add_node.MutableInputDefs()[1];
    NodeArg& a_bias_partition_arg = PartitionWeightByColumn(graph, *a_bias_arg);

    auto b_weight_arg = matmul2_node.MutableInputDefs()[1];
    NodeArg& b_weight_partition_arg = PartitionWeightByRow(graph, *b_weight_arg);

    graph_utils::ReplaceNodeInput(node, 1, a_weight_partition_arg);
    graph_utils::ReplaceNodeInput(add_node, 1, a_bias_partition_arg);
    graph_utils::ReplaceNodeInput(matmul2_node, 1, b_weight_partition_arg);

    const std::vector<NodeArg*> mlp_f_input_defs{node.MutableInputDefs()[0]};
    auto mlp_f_type_info = *node.MutableInputDefs()[0]->TypeAsProto();
    auto& mlp_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MLP_MegatronF_Output"), &mlp_f_type_info);
    Node& mlp_f_node = graph.AddNode(graph.GenerateNodeName("MLP_MegatronF"),
                                     "MegatronF",
                                     "MLP MegatronF",
                                     mlp_f_input_defs,
                                     {&mlp_f_out_arg}, {}, kMSDomain);
    LOGS_DEFAULT(WARNING) << "MLP " << node.Name() << " is Partitioned ";
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
}  // namespace onnxruntime

/*
DenseWeight -- Transpose \
               MatMul -- Relu -- Dropout -- MatMul --Dropout
*/
Status MegatronTransformer::TransformT5MLP(Graph& graph, bool& modified, int graph_level,
                                           const logging::Logger& logger,
                                           std::vector<Node*>& nodes_to_clear_shape,
                                           std::unordered_set<Node*>& self_attention_dropout_nodes, int32_t& counter) const {
  GraphViewer graph_viewer(graph);
  LOGS_DEFAULT(WARNING) << "T5MLP Enter TransformT5MLP";
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {9}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " T5MLP " << node.Name() << " MatMul ";
    Node* second_op = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[1]->Name()));
    Node* first_op = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[0]->Name()));
    if (node.GetInputEdgesCount() > 0) {
      if (second_op == nullptr) {
        LOGS_DEFAULT(WARNING) << " T5MLP " << node.Name() << "'s second input is nullptr";
        break;
      }
      if (first_op != nullptr && first_op->OpType().compare("MegatronF") == 0) {
        continue;
      }

      if (second_op->OpType().compare("Transpose") != 0) {
        continue;
      }
    } else {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " T5MLP " << node.Name() << " Transpose ";
    // todo check tranpose is only 2-dim transpose

    Node* relu_node_ptr = graph.GetNode(node.OutputNodesBegin()->Index());
    Node& relu_node = *relu_node_ptr;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(relu_node, "Relu", {6}) ||
        relu_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        relu_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " T5MLP " << node.Name() << " Relu ";
    Node& dropout_node = *graph.GetNode(relu_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(dropout_node, "Dropout", {12}) ||
        dropout_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        dropout_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " T5MLP " << node.Name() << " Dropout ";
    Node& matmul2_node = *graph.GetNode(dropout_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(matmul2_node, "MatMul", {9}) ||
        matmul2_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        matmul2_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " T5MLP " << node.Name() << " MatMul2 ";
    Node& dropout2_node = *graph.GetNode(matmul2_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(dropout2_node, "Dropout", {12}) ||
        dropout2_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        dropout2_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " T5MLP " << node.Name() << " Dropout2 ";
    Node* transpose_op = const_cast<Node*>(graph.GetProducerNode(matmul2_node.MutableInputDefs()[1]->Name()));
    if (transpose_op->OpType().compare("Transpose") != 0) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " T5MLP " << node.Name() << " Transpose2 ";
    nodes_to_clear_shape.insert(nodes_to_clear_shape.end(), {&node, second_op, &relu_node, &dropout_node,
                                                             &matmul2_node, transpose_op});
    //  &matmul2_node, &dropout2_node, transpose_op});

    auto dense_wi_weight_arg = second_op->MutableInputDefs()[0];
    NodeArg& dense_wi_weight_partition_arg = PartitionWeightByRow(graph, *dense_wi_weight_arg);

    // LOGS_DEFAULT(WARNING) << " T5MLP " << node.Name() << " Partion 1 ";
    auto dense_wo_weight_arg = transpose_op->MutableInputDefs()[0];
    NodeArg& dense_wo_weight_partition_arg = PartitionWeightByColumn(graph, *dense_wo_weight_arg);

    // LOGS_DEFAULT(WARNING) << " T5MLP " << node.Name() << " Partion 2 " << transpose_op->MutableInputDefs()[0]->Name();
    //LOGS_DEFAULT(WARNING) << dense_wo_weight_partition_arg.Name() << "dd" << second_op->MutableInputDefs()[0]->Name()
    //                      << "," << dense_wi_weight_partition_arg.Name();
    graph_utils::ReplaceNodeInput(*second_op, 0, dense_wi_weight_partition_arg);
    graph_utils::ReplaceNodeInput(*transpose_op, 0, dense_wo_weight_partition_arg);

    self_attention_dropout_nodes.insert(&dropout_node);

    const std::vector<NodeArg*> mlp_f_input_defs{node.MutableInputDefs()[0]};
    auto mlp_f_type_info = *node.MutableInputDefs()[0]->TypeAsProto();
    auto& mlp_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MLP_MegatronF_Output"), &mlp_f_type_info);
    Node& mlp_f_node = graph.AddNode(graph.GenerateNodeName("MLP_MegatronF"),
                                     "MegatronF",
                                     "MLP MegatronF",
                                     mlp_f_input_defs,
                                     {&mlp_f_out_arg}, {}, kMSDomain);
    LOGS_DEFAULT(WARNING) << "T5MLP " << node.Name() << " is Partitioned ";
    counter++;
    mlp_f_node.SetExecutionProviderType(node.GetExecutionProviderType());
    const Node::EdgeEnd* edge = graph_utils::GetInputEdge(node, 0);
    if (nullptr == edge) {  // handle input/initializer
      graph_utils::ReplaceNodeInput(node, 0, *(mlp_f_node.MutableOutputDefs()[0]));
    } else {
      auto input_node = const_cast<Node*>(&edge->GetNode());
      graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge->GetSrcArgIndex(), mlp_f_node, 0);
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

/*
DenseWeight -- Transpose \
               MatMul -- BiasGelu -- Dropout -- MatMul -- Add -- Dropout
*/
Status MegatronTransformer::TransformBARTMLP(Graph& graph, bool& modified, int graph_level,
                                             const logging::Logger& logger,
                                             std::vector<Node*>& nodes_to_clear_shape,
                                             std::unordered_set<Node*>& self_attention_dropout_nodes, int32_t& counter) const {
  GraphViewer graph_viewer(graph);
  LOGS_DEFAULT(WARNING) << "BARTMLP Enter TransformBARTMLP";
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {9}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << " MatMul ";
    Node* second_op = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[1]->Name()));
    Node* first_op = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[0]->Name()));
    if (node.GetInputEdgesCount() > 0) {
      if (second_op == nullptr) {
        LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << "'s second input is nullptr";
        break;
      }
      if (first_op != nullptr && first_op->OpType().compare("MegatronF") == 0) {
        continue;
      }

      if (second_op->OpType().compare("Transpose") != 0) {
        continue;
      }
    } else {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << " Transpose ";
    // todo check tranpose is only 2-dim transpose

    Node* biasgelu_node_ptr = graph.GetNode(node.OutputNodesBegin()->Index());
    Node& biasgelu_node = *biasgelu_node_ptr;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(biasgelu_node, "BiasGelu", {1}, kMSDomain) ||
        biasgelu_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        biasgelu_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << " Relu ";
    Node& dropout_node = *graph.GetNode(biasgelu_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(dropout_node, "Dropout", {12}) ||
        dropout_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        dropout_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << " Dropout ";
    Node& matmul2_node = *graph.GetNode(dropout_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(matmul2_node, "MatMul", {9}) ||
        matmul2_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        matmul2_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << " MatMul2 ";
    Node& add_node = *graph.GetNode(matmul2_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7}) ||
        add_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        add_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << " Add "
    Node& dropout2_node = *graph.GetNode(add_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(dropout2_node, "Dropout", {12}) ||
        dropout2_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        dropout2_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << " Dropout2 ";
    Node* transpose_op = const_cast<Node*>(graph.GetProducerNode(matmul2_node.MutableInputDefs()[1]->Name()));
    if (transpose_op->OpType().compare("Transpose") != 0) {
      continue;
    }
    // LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << " Transpose2 ";
    nodes_to_clear_shape.insert(nodes_to_clear_shape.end(), {&node, second_op, &biasgelu_node, &dropout_node,
                                                             &matmul2_node, transpose_op});

    auto dense_wi_weight_arg = second_op->MutableInputDefs()[0];
    NodeArg& dense_wi_weight_partition_arg = PartitionWeightByRow(graph, *dense_wi_weight_arg);

    //since the bias doesnt get transposed, partitioning by col
    auto dense_wi_bias_arg = biasgelu_node.MutableInputDefs()[1];
    NodeArg& dense_wi_bias_partition_arg = PartitionWeightByColumn(graph, *dense_wi_bias_arg);

    // LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << " Partion 1 ";
    auto dense_wo_weight_arg = transpose_op->MutableInputDefs()[0];
    NodeArg& dense_wo_weight_partition_arg = PartitionWeightByColumn(graph, *dense_wo_weight_arg);

    // LOGS_DEFAULT(WARNING) << " BARTMLP " << node.Name() << " Partion 2 " << transpose_op->MutableInputDefs()[0]->Name();
    //LOGS_DEFAULT(WARNING) << dense_wo_weight_partition_arg.Name() << "dd" << second_op->MutableInputDefs()[0]->Name()
    //                      << "," << dense_wi_weight_partition_arg.Name();
    graph_utils::ReplaceNodeInput(*second_op, 0, dense_wi_weight_partition_arg);
    graph_utils::ReplaceNodeInput(biasgelu_node, 1, dense_wi_bias_partition_arg);
    graph_utils::ReplaceNodeInput(*transpose_op, 0, dense_wo_weight_partition_arg);

    self_attention_dropout_nodes.insert(&dropout_node);

    const std::vector<NodeArg*> mlp_f_input_defs{node.MutableInputDefs()[0]};
    auto mlp_f_type_info = *node.MutableInputDefs()[0]->TypeAsProto();
    auto& mlp_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MLP_MegatronF_Output"), &mlp_f_type_info);
    Node& mlp_f_node = graph.AddNode(graph.GenerateNodeName("MLP_MegatronF"),
                                     "MegatronF",
                                     "MLP MegatronF",
                                     mlp_f_input_defs,
                                     {&mlp_f_out_arg}, {}, kMSDomain);
    LOGS_DEFAULT(WARNING) << "BARTMLP " << node.Name() << " is Partitioned ";
    counter++;
    mlp_f_node.SetExecutionProviderType(node.GetExecutionProviderType());
    const Node::EdgeEnd* edge = graph_utils::GetInputEdge(node, 0);
    if (nullptr == edge) {  // handle input/initializer
      graph_utils::ReplaceNodeInput(node, 0, *(mlp_f_node.MutableOutputDefs()[0]));
    } else {
      auto input_node = const_cast<Node*>(&edge->GetNode());
      graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge->GetSrcArgIndex(), mlp_f_node, 0);
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
    Node& k_transpose_after_reshape_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 12];
    Node* matmul_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 11];
    Node* dropout_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 6];
    Node* matmul_node_ptr1 = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 5];
    Node& transpose_node1 = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 4];
    Node& matmul_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 2];

    // Transpose node attribute checking.
    if (!optimizer_utils::IsAttributeWithExpectedValues(k_transpose_after_reshape_node, "perm", {0LL, 2LL, 1LL, 3LL}) ||
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
    NodeArg& qkv_weight_partition_arg = PartitionWeightByColumn(graph, *qkv_weight_arg, 3);

    auto qkv_bias_arg = add_node.MutableInputDefs()[1];
    NodeArg& qkv_bias_partition_arg = PartitionWeightByColumn(graph, *qkv_bias_arg, 3);

    auto dense_weight_arg = matmul_node.MutableInputDefs()[1];
    NodeArg& dense_weight_partition_arg = PartitionWeightByRow(graph, *dense_weight_arg);

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
    graph_utils::ReplaceNodeInput(node, 1, qkv_weight_partition_arg);
    graph_utils::ReplaceNodeInput(add_node, 1, qkv_bias_partition_arg);
    graph_utils::ReplaceNodeInput(matmul_node, 1, dense_weight_partition_arg);

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
    auto& sa_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("SeftAttention_MegatronF_Output"), &sa_f_type_info);
    Node& sa_f_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "SeftAttention_MegatronF"),
                                    "MegatronF",
                                    "SeftAttention MegatronF",
                                    sa_f_input_defs,
                                    {&sa_f_out_arg}, {}, kMSDomain);
    LOGS_DEFAULT(WARNING) << "SeftAttention " << node.Name() << " Partitioned ";
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
    auto& sa_g_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("SeftAttention_MegatronG_Output"), &sa_g_type_info);
    Node& sa_g_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "SelfAttention_MegatronG"),
                                    "MegatronG",
                                    "Attention MegatronG",
                                    sa_g_input_defs,
                                    {&sa_g_out_arg}, {}, kMSDomain);
    sa_g_node.AddAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel));
    sa_g_node.SetExecutionProviderType(node.GetExecutionProviderType());
    graph_utils::ReplaceDownstreamNodeInput(graph, matmul_node, 0, sa_g_node, 0);
    modified = true;
  }

  return Status::OK();
}

Status MegatronTransformer::TransformT5SelfAttention(Graph& graph, bool& modified, int graph_level,
                                                     const logging::Logger& logger,
                                                     std::vector<Node*>& nodes_to_clear_shape,
                                                     std::unordered_set<Node*>& self_attention_dropout_nodes,
                                                     int32_t& counter) const {
  static std::vector<std::string> relative_attention_bias_names;

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  //LOGS_DEFAULT(WARNING) << " Enter T5 Attention ";
  // Self attention sub-graph.
  //
  // MatMul -> Reshape -> Transpose -> MatMul->Add->Softmax->Dropout->MatMul->Transpose->Reshape->MatMul->Droupout
  // MatMul -> Reshape -> Transpose -> |                             |
  // MatMul -> Reshape -> Transpose -------------------------------> |
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", opset_v9) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }
    //{
    //Node* aaaa = const_cast<Node*>(graph.GetProducerNode(std::string("435")));
    //LOGS_DEFAULT(WARNING) << " >>>>>>>>>>>>>>>>>>>>>>>>>>>  because already processed " << (aaaa == nullptr);
    //}

    Node* k_matmul_input_node_ptr = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[0]->Name()));
    if (k_matmul_input_node_ptr != nullptr && k_matmul_input_node_ptr->OpType().compare("MegatronF") == 0) {
      //LOGS_DEFAULT(WARNING) << " Skip T5 Attention " << node.Name() << " because already processed " << node.MutableInputDefs()[0]->Name() << " " << (k_matmul_input_node_ptr == nullptr);
      continue;
    }
    std::vector<Node*> sub_graph_node_ptrs;
    sub_graph_node_ptrs.push_back(&node);
    ProviderType provider_type = node.GetExecutionProviderType();

    std::vector<NodeInfo> linear_pattern = {
        NodeInfo({reshape_info}),
        NodeInfo({transpose_info}),
        NodeInfo({matmul_info}),  // -9
        NodeInfo({add_info}),
        NodeInfo({softmax_info}),
        NodeInfo({dropout_info}, false),  // -6
        NodeInfo({matmul_info}),
        NodeInfo({transpose_info}),
        NodeInfo({reshape_info}),
        NodeInfo({matmul_info}),
        NodeInfo({dropout_info}, false)};  // -1
    if (!MatchLinearPattern(graph, &node, provider_type, linear_pattern, sub_graph_node_ptrs)) {
      continue;
    }
    //LOGS_DEFAULT(WARNING) << " T5 Attention MatchLinearPattern" << node.Name();
    // till now node might be q or k's matmul operation

    // Get all useful nodes here as more vector push back below will change the index.
    Node& k_transpose_after_reshape_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 10];
    Node* matmul_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 9];
    Node* add_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 8];
    //Node* softmax_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 7];
    Node* dropout_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 6];
    Node* matmul_node_ptr1 = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 5];
    Node& transpose_node1 = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 4];
    Node& matmul_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 2];
    //Node& dropout_node2 = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 1];
    // Transpose node attribute checking.
    if (!optimizer_utils::IsAttributeWithExpectedValues(k_transpose_after_reshape_node, "perm", {0LL, 2LL, 3LL, 1LL}) ||
        !optimizer_utils::IsAttributeWithExpectedValues(transpose_node1, "perm", {0LL, 2LL, 1LL, 3LL})) {
      continue;
    }

    //std::vector<Node*> transpose_node_ptrs;  // For the k and v matmul transpose nodes.
    std::vector<Node*> reshape_node_ptrs;  // To keep the reshape node that need to change the shape constant.
    reshape_node_ptrs.push_back(sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 11]);
    reshape_node_ptrs.push_back(sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 3]);
    //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check" << node.Name();
    // till now node should be k matmul operation

    std::vector<Node*> weight_transpose_node_ptrs;
    Node* k_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[1]->Name()));
    ORT_ENFORCE(k_transpose_ptr->OpType().compare("Transpose") == 0);
    weight_transpose_node_ptrs.push_back(k_transpose_ptr);
    sub_graph_node_ptrs.push_back(k_transpose_ptr);

    Node* q_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(matmul_node_ptr->MutableInputDefs()[0]->Name()));
    ORT_ENFORCE(q_transpose_ptr->OpType().compare("Transpose") == 0);
    sub_graph_node_ptrs.push_back(q_transpose_ptr);
    Node* q_reshape = const_cast<Node*>(graph.GetProducerNode(q_transpose_ptr->MutableInputDefs()[0]->Name()));
    reshape_node_ptrs.push_back(q_reshape);
    sub_graph_node_ptrs.push_back(q_reshape);
    Node* q_matmul = const_cast<Node*>(graph.GetProducerNode(q_reshape->MutableInputDefs()[0]->Name()));
    sub_graph_node_ptrs.push_back(q_matmul);
    Node* q_weight_transpose = const_cast<Node*>(graph.GetProducerNode(q_matmul->MutableInputDefs()[1]->Name()));
    sub_graph_node_ptrs.push_back(q_weight_transpose);
    weight_transpose_node_ptrs.push_back(q_weight_transpose);

    Node* v_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(matmul_node_ptr1->MutableInputDefs()[1]->Name()));
    ORT_ENFORCE(v_transpose_ptr != nullptr);
    ORT_ENFORCE(v_transpose_ptr->OpType().compare("Transpose") == 0);
    sub_graph_node_ptrs.push_back(v_transpose_ptr);
    Node* v_reshape = const_cast<Node*>(graph.GetProducerNode(v_transpose_ptr->MutableInputDefs()[0]->Name()));
    ORT_ENFORCE(v_reshape != nullptr);
    reshape_node_ptrs.push_back(v_reshape);
    sub_graph_node_ptrs.push_back(v_reshape);
    Node* v_matmul = const_cast<Node*>(graph.GetProducerNode(v_reshape->MutableInputDefs()[0]->Name()));
    sub_graph_node_ptrs.push_back(v_matmul);
    ORT_ENFORCE(v_matmul != nullptr);
    Node* v_weight_transpose = const_cast<Node*>(graph.GetProducerNode(v_matmul->MutableInputDefs()[1]->Name()));
    ORT_ENFORCE(v_weight_transpose != nullptr);
    sub_graph_node_ptrs.push_back(v_weight_transpose);
    weight_transpose_node_ptrs.push_back(v_weight_transpose);

    // slice relative attention  bias
    Node* add_input_ptr = const_cast<Node*>(graph.GetProducerNode(add_node_ptr->MutableInputDefs()[1]->Name()));
    ORT_ENFORCE(add_input_ptr != nullptr);
    sub_graph_node_ptrs.push_back(add_input_ptr);
    Node* unsqueeze_ptr = const_cast<Node*>(graph.GetProducerNode(add_input_ptr->MutableInputDefs()[0]->Name()));
    sub_graph_node_ptrs.push_back(unsqueeze_ptr);
    ORT_ENFORCE(unsqueeze_ptr != nullptr);
    Node* rab_tranpose_ptr = const_cast<Node*>(graph.GetProducerNode(unsqueeze_ptr->MutableInputDefs()[0]->Name()));
    sub_graph_node_ptrs.push_back(rab_tranpose_ptr);
    ORT_ENFORCE(rab_tranpose_ptr != nullptr);
    Node* rab_gather_ptr = const_cast<Node*>(graph.GetProducerNode(rab_tranpose_ptr->MutableInputDefs()[0]->Name()));
    sub_graph_node_ptrs.push_back(rab_gather_ptr);
    ORT_ENFORCE(rab_gather_ptr != nullptr);

    //LOGS_DEFAULT(WARNING) << " T5 Attention 99999999" << node.Name() << weight_transpose_node_ptrs.size();
    // Sub-graph structure and transpose attribute checking.
    //if (transpose_node_ptrs.size() != 2 ||
    //    !optimizer_utils::IsAttributeWithExpectedValues(*transpose_node_ptrs[0], "perm", {0LL, 2LL, 3LL, 1LL}) ||
    //    !optimizer_utils::IsAttributeWithExpectedValues(*transpose_node_ptrs[1], "perm", {0LL, 2LL, 1LL, 3LL})) {
    //  continue;
    //}

    // K and V matmul must have the same input
    Node* k_matmul = &node;
    ORT_ENFORCE(k_matmul->MutableInputDefs()[0]->Name() == v_matmul->MutableInputDefs()[0]->Name());

    bool need_skip = false;
    for (auto trans_ptr : weight_transpose_node_ptrs) {
      const ONNX_NAMESPACE::TensorProto* tensor_proto;
      if (!graph.GetInitializedTensor(trans_ptr->MutableInputDefs()[0]->Name(), tensor_proto)) {
        //LOGS_DEFAULT(WARNING) << " T5 Attention Skipping now" << trans_ptr->MutableInputDefs()[0]->Name() << " " << trans_ptr->Name();
        need_skip = true;
        break;
      }
    }
    if (need_skip) {
      //LOGS_DEFAULT(WARNING) << " T5 Attention Skipp because transposes' inputs are not initializers" << node.Name();
      continue;
    }

    //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check 333" << node.Name();
    //transpose_node_ptrs.push_back(&k_transpose_after_reshape_node);
    // Partition weights. If any of them fails, skip transforming this sub-graph.
    for (auto trans_ptr : weight_transpose_node_ptrs) {
      auto qkv_weight_arg = trans_ptr->MutableInputDefs()[0];
      //LOGS_DEFAULT(WARNING) << " T5 Attention Weight Transpose Loop " << qkv_weight_arg->Name();
      NodeArg& qkv_weight_partition_arg = PartitionWeightByRow(graph, *qkv_weight_arg);
      graph_utils::ReplaceNodeInput(*trans_ptr, 0, qkv_weight_partition_arg);
      //LOGS_DEFAULT(WARNING) << " T5 Attention Weight Transpose Loop " << qkv_weight_partition_arg.Name();
      //ORT_ENFORCE(qkv_weight_arg == &qkv_weight_partition_arg);
    }
    //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check 444444" << node.Name();
    Node* last_transpose = const_cast<Node*>(graph.GetProducerNode(matmul_node.MutableInputDefs()[1]->Name()));
    auto dense_weight_arg = last_transpose->MutableInputDefs()[0];
    NodeArg& dense_weight_partition_arg = PartitionWeightByColumn(graph, *dense_weight_arg);
    graph_utils::ReplaceNodeInput(*last_transpose, 0, dense_weight_partition_arg);
    //ORT_ENFORCE(dense_weight_arg == &dense_weight_partition_arg);
    //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check 55555" << node.Name();

    auto rab_weight_arg = rab_gather_ptr->MutableInputDefs()[0];
    //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check 555509999995" << rab_weight_arg->Name();
    if (std::find(relative_attention_bias_names.begin(), relative_attention_bias_names.end(), rab_weight_arg->Name()) == relative_attention_bias_names.end()) {
      // LOGS_DEFAULT(WARNING) << " T5 Attention Partitoned Relative Attention " << rab_weight_arg->Name();
      NodeArg& rab_weight_partition_arg = PartitionWeightByColumn(graph, *rab_weight_arg);
      graph_utils::ReplaceNodeInput(*rab_gather_ptr, 0, rab_weight_partition_arg);
      //ORT_ENFORCE(rab_weight_arg == &rab_weight_partition_arg);
      relative_attention_bias_names.push_back(rab_weight_partition_arg.Name());
    } else {
      //LOGS_DEFAULT(WARNING) << " Skip T5 Attention Partitoned Relative Attention because already partitioned " << rab_weight_arg->Name();
    }

    // Check the constant value in the Reshape nodes.
    bool is_reshape_valid = true;
    for (Node* node_ptr : reshape_node_ptrs) {
      //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check 565656565" << node_ptr->Name();
      auto shape_arg = node_ptr->MutableInputDefs()[1];
      const ONNX_NAMESPACE::TensorProto* tensor;
      //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check 6666" << node.Name();
      if (!graph.GetInitializedTensor(shape_arg->Name(), tensor)) {
        is_reshape_valid = false;
        break;
      }
      //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check 7777" << node.Name();
      auto data_type = tensor->data_type();
      if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        is_reshape_valid = false;
        break;
      }
      //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check 8888" << node.Name();
      // The number of the values should be more than 2, and the 3rd value should be divisible by parallel size,
      // i.e., the attention head number should be divisible by parallel size.
      auto init_const = onnxruntime::make_unique<Initializer>(*tensor, graph.ModelPath());
      if (init_const->size() != 3 && init_const->size() != 4) {
        is_reshape_valid = false;
        break;
      }
      //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check 99999" << node.Name();
      const int64_t* val = init_const->data<int64_t>();
      if (val[2] % horizontal_parallel_size_ != 0) {
        LOGS_DEFAULT(WARNING) << (init_const->size() == 3 ? "Hidden size " : "Number of attention heads ") << val[2]
                              << " is not divisible by horizontal_parallel_size_ "
                              << horizontal_parallel_size_ << ", not supported currently.";
        is_reshape_valid = false;
        break;
      }
      //LOGS_DEFAULT(WARNING) << " T5 Attention Transpose Check 10101010101" << node.Name();
    }

    if (!is_reshape_valid) {
      continue;
    }
    //LOGS_DEFAULT(WARNING) << " T5 Attention Reshape Check" << node.Name();
    // Ready to transform the sub-graph when reach here.
    // It's possible that the node vector contains nullptr due to some optinal node infos during linear pattern matching.
    std::copy_if(sub_graph_node_ptrs.begin(), sub_graph_node_ptrs.end(),
                 std::back_inserter(nodes_to_clear_shape),
                 [](Node* node_ptr) { return node_ptr != nullptr; });

    // Replace by the partition weights.
    //graph_utils::ReplaceNodeInput(node, 1, qkv_weight_partition_arg);
    //graph_utils::ReplaceNodeInput(add_node, 1, qkv_bias_partition_arg);
    //graph_utils::ReplaceNodeInput(matmul_node, 1, dense_weight_partition_arg);

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
      //LOGS_DEFAULT(WARNING) << "Replace Reshape node's shape " << shape_arg->Name() << " " << val_partition[0] << "," << val_partition[1] << "," << val_partition[2];
      tensor_partition.set_raw_data(val_partition.data(), size * sizeof(int64_t));
      NodeArg& node_arg_partition = graph_utils::AddInitializer(graph, tensor_partition);
      graph_utils::ReplaceNodeInput(*node_ptr, 1, node_arg_partition);
      graph.RemoveInitializedTensor(shape_arg->Name());
    }

    if (dropout_node_ptr != nullptr) {
      self_attention_dropout_nodes.insert(dropout_node_ptr);
    }

    // Add MegatronF before the 1st MatMul and MegatronG before the last Dropout.

    NodeArg* prev_input_node_ptr = node.MutableInputDefs()[0];
    std::vector<Node*> new_consumer_nodes;
    const auto& node_consumers = graph.GetConsumerNodes(prev_input_node_ptr->Name());
    for (auto& n : node_consumers) {
      if (n->Index() == k_matmul->Index() || n->Index() == v_matmul->Index() || n->Index() == q_matmul->Index()) {
        continue;
      }
      new_consumer_nodes.emplace_back(const_cast<Node*>(n));
    }

    bool shared_same_input = k_matmul->MutableInputDefs()[0]->Name().compare(q_matmul->MutableInputDefs()[0]->Name()) == 0;

    //then for q, and k&v will have different MegatronF node.
    {
      const std::vector<NodeArg*> sa_f_input_defs{prev_input_node_ptr};
      auto sa_f_type_info = *prev_input_node_ptr->TypeAsProto();
      auto& sa_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(k_matmul->Name() + "T5Attention_MegatronF_Output"), &sa_f_type_info);
      Node& sa_f_node = graph.AddNode(graph.GenerateNodeName(k_matmul->Name() + "T5Attention_MegatronF"),
                                      "MegatronF",
                                      k_matmul->Name() + "T5Attention MegatronF",
                                      sa_f_input_defs,
                                      {&sa_f_out_arg}, {}, kMSDomain);
      sa_f_node.SetExecutionProviderType(k_matmul->GetExecutionProviderType());
      graph_utils::ReplaceNodeInput(node, 0, *(sa_f_node.MutableOutputDefs()[0]));
      graph_utils::ReplaceNodeInput(*v_matmul, 0, *(sa_f_node.MutableOutputDefs()[0]));
      if (shared_same_input) {
        graph_utils::ReplaceNodeInput(*q_matmul, 0, *(sa_f_node.MutableOutputDefs()[0]));
      }
      //     const Node::EdgeEnd* edge = graph_utils::GetInputEdge(node, 0);
      // if (nullptr == edge) {  // handle input/initializer
      //   graph_utils::ReplaceNodeInput(node, 0, *(sa_f_node.MutableOutputDefs()[0]));
      // } else {
      //   auto input_node = const_cast<Node*>(&edge->GetNode());
      //   graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge->GetDstArgIndex(), sa_f_node, 0);
      // }
      new_consumer_nodes.push_back(&sa_f_node);
    }
    graph.UpdateConsumerNodes(prev_input_node_ptr->Name(), new_consumer_nodes);
    LOGS_DEFAULT(WARNING) << "T5 Attention " << k_matmul->Name() << " Partitioned " << k_matmul->MutableInputDefs()[0]->Name();
    counter++;
    if (!shared_same_input) {
      {
        NodeArg* prev_input_node_ptr = q_matmul->MutableInputDefs()[0];
        std::vector<Node*> new_consumer_nodes;
        const auto& node_consumers = graph.GetConsumerNodes(prev_input_node_ptr->Name());
        for (auto& n : node_consumers) {
          if (n->Index() == k_matmul->Index() || n->Index() == v_matmul->Index() || n->Index() == q_matmul->Index()) {
            continue;
          }
          new_consumer_nodes.emplace_back(const_cast<Node*>(n));
        }

        const std::vector<NodeArg*> sa_f_input_defs{q_matmul->MutableInputDefs()[0]};
        auto sa_f_type_info = *q_matmul->MutableInputDefs()[0]->TypeAsProto();
        auto& sa_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "T5Attention_MegatronF_Output"), &sa_f_type_info);
        Node& sa_f_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "T5Attention_MegatronF"),
                                        "MegatronF",
                                        node.Name() + "T5Attention MegatronF",
                                        sa_f_input_defs,
                                        {&sa_f_out_arg}, {}, kMSDomain);
        sa_f_node.SetExecutionProviderType(q_matmul->GetExecutionProviderType());

        graph_utils::ReplaceNodeInput(*q_matmul, 0, *(sa_f_node.MutableOutputDefs()[0]));
        // const Node::EdgeEnd* edge = graph_utils::GetInputEdge(*q_matmul, 0);
        // if (nullptr == edge) {  // handle input/initializer
        //   graph_utils::ReplaceNodeInput(*q_matmul, 0, *(sa_f_node.MutableOutputDefs()[0]));
        // } else {
        //   auto input_node = const_cast<Node*>(&edge->GetNode());
        //   graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge->GetDstArgIndex(), sa_f_node, 0);
        // }
        new_consumer_nodes.push_back(&sa_f_node);
        graph.UpdateConsumerNodes(prev_input_node_ptr->Name(), new_consumer_nodes);
        // todo: need update the consumer node for the input_node as well.
      }
    }

    const std::vector<NodeArg*> sa_g_input_defs{matmul_node.MutableOutputDefs()[0]};
    auto sa_g_type_info = *matmul_node.MutableOutputDefs()[0]->TypeAsProto();  // copy
    //LOGS_DEFAULT(WARNING) << "%%%%%%%%%%%%%%%%%%%%%" << node.Name() << "dim 0: "
    //                      << sa_g_type_info.tensor_type().shape().dim(0).dim_value() << ", dim 1:"
    //                      << sa_g_type_info.tensor_type().shape().dim(1).dim_value();
    auto& sa_g_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("T5Attention_MegatronG_Output"), &sa_g_type_info);
    Node& sa_g_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "T5Attention_MegatronG"),
                                    "MegatronG",
                                    "T5Attention MegatronG",
                                    sa_g_input_defs,
                                    {&sa_g_out_arg}, {}, kMSDomain);
    sa_g_node.AddAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel));
    sa_g_node.SetExecutionProviderType(node.GetExecutionProviderType());
    graph_utils::ReplaceDownstreamNodeInput(graph, matmul_node, 0, sa_g_node, 0);

    // // Add MegatronF before the relative attention bias add.
    // const std::vector<NodeArg*> sa_f_input_defs2{add_node_ptr->MutableInputDefs()[0]};
    // auto sa_f_type_info2 = *add_node_ptr->MutableInputDefs()[0]->TypeAsProto();
    // auto& sa_f_out_arg2 = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("SelfAttention_MegatronF_Output"), &sa_f_type_info2);
    // Node& sa_f_node2 = graph.AddNode(graph.GenerateNodeName("SelfAttention_MegatronF"),
    //                                  "MegatronF",
    //                                  "SelfAttention MegatronF",
    //                                  sa_f_input_defs2,
    //                                  {&sa_f_out_arg2}, {}, kMSDomain);
    // sa_f_node2.SetExecutionProviderType(node.GetExecutionProviderType());
    // const Node::EdgeEnd* edge2 = graph_utils::GetInputEdge(*add_node_ptr, 0);
    // if (nullptr == edge2) {  // handle input/initializer
    //   graph_utils::ReplaceNodeInput(*add_node_ptr, 0, *(sa_f_node2.MutableOutputDefs()[0]));
    // } else {
    //   auto input_node = const_cast<Node*>(&edge2->GetNode());
    //   graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge2->GetSrcArgIndex(), sa_f_node2, 0);
    // }

    modified = true;
  }

  return Status::OK();
}

Status MegatronTransformer::TransformBARTSelfAttention(Graph& graph, bool& modified, int graph_level,
                                                       const logging::Logger& logger,
                                                       std::vector<Node*>& nodes_to_clear_shape,
                                                       std::unordered_set<Node*>& self_attention_dropout_nodes,
                                                       int32_t& counter) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  LOGS_DEFAULT(WARNING) << " Enter BART Attention ";

  // Self attention sub-graph.
  //
  // MatMul->Add->Mul->Reshape->Transpose->MatMul->Reshape->Where->Reshape->Softmax->Dropout->MatMul->Transpose->Reshape->MatMul->Add->Droupout
  // MatMul->Add->Reshape->Transpose-------> |                                                  |
  // MatMul->Add->Reshape->Transpose----------------------------------------------------------> |
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", opset_v9) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    Node* k_matmul_input_node_ptr = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[0]->Name()));
    if (k_matmul_input_node_ptr != nullptr && k_matmul_input_node_ptr->OpType().compare("MegatronF") == 0) {
      continue;
    }
    std::vector<Node*> sub_graph_node_ptrs;
    sub_graph_node_ptrs.push_back(&node);
    ProviderType provider_type = node.GetExecutionProviderType();

    std::vector<NodeInfo> linear_pattern = {
        NodeInfo({add_info}),
        NodeInfo({mul_info}),
        NodeInfo({reshape_info}),
        NodeInfo({transpose_info}),
        NodeInfo({matmul_info}),
        NodeInfo({add_info}, false),  // -13
        NodeInfo({reshape_info}),
        NodeInfo({where_info}),
        NodeInfo({reshape_info}),
        NodeInfo({softmax_info}),
        NodeInfo({dropout_info}, false),  // -8
        NodeInfo({matmul_info}),
        NodeInfo({add_info}, false),
        NodeInfo({transpose_info}),
        NodeInfo({reshape_info}),
        NodeInfo({matmul_info}),  // -3
        NodeInfo({add_info}),
        NodeInfo({dropout_info}, false)};  // -1
    if (!MatchLinearPattern(graph, &node, provider_type, linear_pattern, sub_graph_node_ptrs)) {
      continue;
    }
    LOGS_DEFAULT(WARNING) << " BART Attention: linear pattern match. ";
    // Get all useful nodes here as more vector push back below will change the index.
    Node* q_biasadd_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 18];
    Node& q_transpose_after_reshape_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 15];
    Node* qk_matmul_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 14];
    Node* dropout_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 8];
    Node* qkv_matmul_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 7];
    Node& transpose_node1 = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 5];
    Node& dense_matmul_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 3];

    // Transpose node attribute checking.
    if (!optimizer_utils::IsAttributeWithExpectedValues(q_transpose_after_reshape_node, "perm", {1LL, 0LL, 2LL}) ||
        !optimizer_utils::IsAttributeWithExpectedValues(transpose_node1, "perm", {1LL, 0LL, 2LL})) {
      continue;
    }
    LOGS_DEFAULT(WARNING) << " BART Attention: transpose attribute match. ";
    //std::vector<Node*> transpose_node_ptrs;  // For the k and v matmul transpose nodes.
    // std::vector<Node*> reshape_node_ptrs;  // To keep the reshape node that need to change the shape constant.
    std::unordered_map<Node*, int64_t> reshape_node_ptrs;
    reshape_node_ptrs[sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 16]] = 1;
    reshape_node_ptrs[sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 12]] = 1;  //dont need change
    reshape_node_ptrs[sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 10]] = 0;  //dont need change
    reshape_node_ptrs[sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 4]] = 2;
    // till now node should be k matmul operation

    std::vector<Node*> weight_transpose_node_ptrs;
    std::vector<Node*> bias_add_node_ptrs;

    Node* q_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[1]->Name()));
    ORT_ENFORCE(q_transpose_ptr->OpType().compare("Transpose") == 0);
    weight_transpose_node_ptrs.push_back(q_transpose_ptr);
    sub_graph_node_ptrs.push_back(q_transpose_ptr);
    bias_add_node_ptrs.push_back(q_biasadd_node_ptr);

    LOGS_DEFAULT(WARNING) << " BART Attention: q info done. ";

    Node* k_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(qk_matmul_node_ptr->MutableInputDefs()[1]->Name()));
    ORT_ENFORCE(k_transpose_ptr->OpType().compare("Transpose") == 0);
    sub_graph_node_ptrs.push_back(k_transpose_ptr);
    Node* k_reshape = const_cast<Node*>(graph.GetProducerNode(k_transpose_ptr->MutableInputDefs()[0]->Name()));
    reshape_node_ptrs[k_reshape] = 1;
    sub_graph_node_ptrs.push_back(k_reshape);
    // specific to BART, there is a lingering transpose, need to debug but temp code:
    // if(k_reshape->GetOutputEdgesCount() == 2)
    // {
    //   Node& tp0_ptr = graph.GetNode(k_reshape->OutputNodesBegin()->Index());
    //   Node& tp1_ptr = graph.GetNode(k_reshape->OutputNodesEnd()->Index());
    //   if(tp0.Name() == k_transpose_ptr->Name()){
    //     sub_graph_node_ptrs.push_back(&tp1);
    //     LOGS_DEFAULT(WARNING) << " BART Attention: removing shape of . "<< tp1.Name();
    //   }
    //   else if(tp1.Name() == k_transpose_ptr->Name()){
    //     sub_graph_node_ptrs.push_back(&tp0);
    //     LOGS_DEFAULT(WARNING) << " BART Attention: removing shape of . "<< tp0.Name();
    //   }
    // }
    // Node* k_mul = const_cast<Node*>(graph.GetProducerNode(k_reshape->MutableInputDefs()[0]->Name()));
    // sub_graph_node_ptrs.push_back(k_mul);
    Node* k_add = const_cast<Node*>(graph.GetProducerNode(k_reshape->MutableInputDefs()[0]->Name()));
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*k_add, "Add", {7})) {
      continue;
    }
    sub_graph_node_ptrs.push_back(k_add);
    bias_add_node_ptrs.push_back(k_add);
    Node* k_matmul = const_cast<Node*>(graph.GetProducerNode(k_add->MutableInputDefs()[0]->Name()));
    sub_graph_node_ptrs.push_back(k_matmul);
    Node* k_weight_transpose = const_cast<Node*>(graph.GetProducerNode(k_matmul->MutableInputDefs()[1]->Name()));
    sub_graph_node_ptrs.push_back(k_weight_transpose);
    weight_transpose_node_ptrs.push_back(k_weight_transpose);
    LOGS_DEFAULT(WARNING) << " BART Attention: k info done. ";

    Node* v_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(qkv_matmul_node_ptr->MutableInputDefs()[1]->Name()));
    ORT_ENFORCE(v_transpose_ptr != nullptr);
    ORT_ENFORCE(v_transpose_ptr->OpType().compare("Transpose") == 0);
    sub_graph_node_ptrs.push_back(v_transpose_ptr);
    Node* v_reshape = const_cast<Node*>(graph.GetProducerNode(v_transpose_ptr->MutableInputDefs()[0]->Name()));
    ORT_ENFORCE(v_reshape != nullptr);
    reshape_node_ptrs[v_reshape] = 1;
    sub_graph_node_ptrs.push_back(v_reshape);
    Node* v_add = const_cast<Node*>(graph.GetProducerNode(v_reshape->MutableInputDefs()[0]->Name()));
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*v_add, "Add", {7})) {
      continue;
    }
    sub_graph_node_ptrs.push_back(v_add);
    bias_add_node_ptrs.push_back(v_add);
    ORT_ENFORCE(v_add != nullptr);
    Node* v_matmul = const_cast<Node*>(graph.GetProducerNode(v_add->MutableInputDefs()[0]->Name()));
    sub_graph_node_ptrs.push_back(v_matmul);
    ORT_ENFORCE(v_matmul != nullptr);
    Node* v_weight_transpose = const_cast<Node*>(graph.GetProducerNode(v_matmul->MutableInputDefs()[1]->Name()));
    ORT_ENFORCE(v_weight_transpose != nullptr);
    sub_graph_node_ptrs.push_back(v_weight_transpose);
    weight_transpose_node_ptrs.push_back(v_weight_transpose);
    LOGS_DEFAULT(WARNING) << " BART Attention: v info done. ";

    // K and V matmul must have the same input
    Node* q_matmul = &node;
    ORT_ENFORCE(k_matmul->MutableInputDefs()[0]->Name() == v_matmul->MutableInputDefs()[0]->Name());

    bool need_skip = false;
    for (auto trans_ptr : weight_transpose_node_ptrs) {
      const ONNX_NAMESPACE::TensorProto* tensor_proto;
      if (!graph.GetInitializedTensor(trans_ptr->MutableInputDefs()[0]->Name(), tensor_proto)) {
        LOGS_DEFAULT(WARNING) << " BART Attention Skipping now" << trans_ptr->MutableInputDefs()[0]->Name() << " " << trans_ptr->Name();
        need_skip = true;
        break;
      }
    }
    if (need_skip) {
      LOGS_DEFAULT(WARNING) << " BART Attention Skipp because transposes' inputs are not initializers" << node.Name();
      continue;
    }

    // Check the constant value in the Reshape nodes.
    bool is_reshape_valid = true;
    for (auto x : reshape_node_ptrs) {
      Node* node_ptr = x.first;
      int64_t idx = x.second;
      LOGS_DEFAULT(WARNING) << " BART Attention Transpose Check 565656565" << node_ptr->Name();
      auto shape_arg = node_ptr->MutableInputDefs()[1];
      const ONNX_NAMESPACE::TensorProto* tensor;
      LOGS_DEFAULT(WARNING) << " BART Attention Transpose Check 6666" << node.Name();
      if (!graph.GetInitializedTensor(shape_arg->Name(), tensor)) {
        is_reshape_valid = false;
        break;
      }
      LOGS_DEFAULT(WARNING) << " BART Attention Transpose Check 7777" << node.Name();
      auto data_type = tensor->data_type();
      if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        is_reshape_valid = false;
        break;
      }
      LOGS_DEFAULT(WARNING) << " BART Attention Transpose Check 8888" << node.Name();
      // The number of the values should be more than 2, and the 3rd value should be divisible by parallel size,
      // i.e., the attention head number should be divisible by parallel size.
      auto init_const = onnxruntime::make_unique<Initializer>(*tensor, graph.ModelPath());
      if (init_const->size() <= idx) {
        is_reshape_valid = false;
        break;
      }
      LOGS_DEFAULT(WARNING) << " BART Attention Transpose Check 99999" << node.Name();
      const int64_t* val = init_const->data<int64_t>();
      // if (val[2] % horizontal_parallel_size_ != 0) {
      if (val[idx] % horizontal_parallel_size_ != 0) {
        LOGS_DEFAULT(WARNING) << "dim[" << idx << "]: " << val[idx]
                              << " is not divisible by horizontal_parallel_size_ "
                              << horizontal_parallel_size_ << ", not supported currently.";
        is_reshape_valid = false;
        break;
      }
      LOGS_DEFAULT(WARNING) << " BART Attention Transpose Check 10101010101" << node.Name();
    }

    if (!is_reshape_valid) {
      LOGS_DEFAULT(WARNING) << " BART Attention:reshape invalid, exiting ";
      continue;
    }

    // Partition weights. If any of them fails, skip transforming this sub-graph.
    for (auto trans_ptr : weight_transpose_node_ptrs) {
      auto qkv_weight_arg = trans_ptr->MutableInputDefs()[0];
      NodeArg& qkv_weight_partition_arg = PartitionWeightByRow(graph, *qkv_weight_arg);
      graph_utils::ReplaceNodeInput(*trans_ptr, 0, qkv_weight_partition_arg);
    }

    // Partition bias. If any of them fails, skip transforming this sub-graph.
    for (auto add_ptr : bias_add_node_ptrs) {
      auto qkv_bias_arg = add_ptr->MutableInputDefs()[1];
      NodeArg& qkv_bias_partition_arg = PartitionWeightByColumn(graph, *qkv_bias_arg);
      graph_utils::ReplaceNodeInput(*add_ptr, 1, qkv_bias_partition_arg);
    }

    Node* last_transpose = const_cast<Node*>(graph.GetProducerNode(dense_matmul_node.MutableInputDefs()[1]->Name()));
    auto dense_weight_arg = last_transpose->MutableInputDefs()[0];
    NodeArg& dense_weight_partition_arg = PartitionWeightByColumn(graph, *dense_weight_arg);
    graph_utils::ReplaceNodeInput(*last_transpose, 0, dense_weight_partition_arg);

    //LOGS_DEFAULT(WARNING) << " BART Attention Reshape Check" << node.Name();
    // Ready to transform the sub-graph when reach here.
    // It's possible that the node vector contains nullptr due to some optinal node infos during linear pattern matching.
    std::copy_if(sub_graph_node_ptrs.begin(), sub_graph_node_ptrs.end(),
                 std::back_inserter(nodes_to_clear_shape),
                 [](Node* node_ptr) { return node_ptr != nullptr; });

    // Change the constant for the reshape nodes.
    for (auto x : reshape_node_ptrs) {
      Node* node_ptr = x.first;
      int64_t idx = x.second;
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
      // val_partition[2] /= horizontal_parallel_size_;
      val_partition[idx] /= horizontal_parallel_size_;
      //LOGS_DEFAULT(WARNING) << "Replace Reshape node's shape " << shape_arg->Name() << " " << val_partition[0] << "," << val_partition[1] << "," << val_partition[2];
      tensor_partition.set_raw_data(val_partition.data(), size * sizeof(int64_t));
      NodeArg& node_arg_partition = graph_utils::AddInitializer(graph, tensor_partition);
      graph_utils::ReplaceNodeInput(*node_ptr, 1, node_arg_partition);
      graph.RemoveInitializedTensor(shape_arg->Name());
    }

    if (dropout_node_ptr != nullptr) {
      self_attention_dropout_nodes.insert(dropout_node_ptr);
    }

    // Add MegatronF before the 1st MatMul and MegatronG before the last Add.

    NodeArg* prev_input_node_ptr = k_matmul->MutableInputDefs()[0];
    std::vector<Node*> new_consumer_nodes;
    const auto& node_consumers = graph.GetConsumerNodes(prev_input_node_ptr->Name());
    for (auto& n : node_consumers) {
      if (n->Index() == k_matmul->Index() || n->Index() == v_matmul->Index() || n->Index() == q_matmul->Index()) {
        continue;
      }
      new_consumer_nodes.emplace_back(const_cast<Node*>(n));
    }

    bool shared_same_input = k_matmul->MutableInputDefs()[0]->Name().compare(q_matmul->MutableInputDefs()[0]->Name()) == 0;

    //then for q, and k&v will have different MegatronF node.
    {
      const std::vector<NodeArg*> sa_f_input_defs{prev_input_node_ptr};
      auto sa_f_type_info = *prev_input_node_ptr->TypeAsProto();
      auto& sa_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(k_matmul->Name() + "BARTAttention_MegatronF_Output"), &sa_f_type_info);
      Node& sa_f_node = graph.AddNode(graph.GenerateNodeName(k_matmul->Name() + "BARTAttention_MegatronF"),
                                      "MegatronF",
                                      k_matmul->Name() + "BARTAttention MegatronF",
                                      sa_f_input_defs,
                                      {&sa_f_out_arg}, {}, kMSDomain);
      sa_f_node.SetExecutionProviderType(k_matmul->GetExecutionProviderType());
      graph_utils::ReplaceNodeInput(*k_matmul, 0, *(sa_f_node.MutableOutputDefs()[0]));
      graph_utils::ReplaceNodeInput(*v_matmul, 0, *(sa_f_node.MutableOutputDefs()[0]));
      if (shared_same_input) {
        graph_utils::ReplaceNodeInput(*q_matmul, 0, *(sa_f_node.MutableOutputDefs()[0]));
      }
      //     const Node::EdgeEnd* edge = graph_utils::GetInputEdge(node, 0);
      // if (nullptr == edge) {  // handle input/initializer
      //   graph_utils::ReplaceNodeInput(node, 0, *(sa_f_node.MutableOutputDefs()[0]));
      // } else {
      //   auto input_node = const_cast<Node*>(&edge->GetNode());
      //   graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge->GetDstArgIndex(), sa_f_node, 0);
      // }
      new_consumer_nodes.push_back(&sa_f_node);
    }
    graph.UpdateConsumerNodes(prev_input_node_ptr->Name(), new_consumer_nodes);
    LOGS_DEFAULT(WARNING) << "BART Attention " << k_matmul->Name() << " Partitioned " << k_matmul->MutableInputDefs()[0]->Name();
    counter++;
    if (!shared_same_input) {
      {
        NodeArg* prev_input_node_ptr = q_matmul->MutableInputDefs()[0];
        std::vector<Node*> new_consumer_nodes;
        const auto& node_consumers = graph.GetConsumerNodes(prev_input_node_ptr->Name());
        for (auto& n : node_consumers) {
          if (n->Index() == k_matmul->Index() || n->Index() == v_matmul->Index() || n->Index() == q_matmul->Index()) {
            continue;
          }
          new_consumer_nodes.emplace_back(const_cast<Node*>(n));
        }

        const std::vector<NodeArg*> sa_f_input_defs{q_matmul->MutableInputDefs()[0]};
        auto sa_f_type_info = *q_matmul->MutableInputDefs()[0]->TypeAsProto();
        auto& sa_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(q_matmul->Name() + "BARTAttention_MegatronF_Output"), &sa_f_type_info);
        Node& sa_f_node = graph.AddNode(graph.GenerateNodeName(q_matmul->Name() + "BARTAttention_MegatronF"),
                                        "MegatronF",
                                        q_matmul->Name() + "BARTAttention MegatronF",
                                        sa_f_input_defs,
                                        {&sa_f_out_arg}, {}, kMSDomain);
        sa_f_node.SetExecutionProviderType(q_matmul->GetExecutionProviderType());

        graph_utils::ReplaceNodeInput(*q_matmul, 0, *(sa_f_node.MutableOutputDefs()[0]));
        // const Node::EdgeEnd* edge = graph_utils::GetInputEdge(*q_matmul, 0);
        // if (nullptr == edge) {  // handle input/initializer
        //   graph_utils::ReplaceNodeInput(*q_matmul, 0, *(sa_f_node.MutableOutputDefs()[0]));
        // } else {
        //   auto input_node = const_cast<Node*>(&edge->GetNode());
        //   graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge->GetDstArgIndex(), sa_f_node, 0);
        // }
        new_consumer_nodes.push_back(&sa_f_node);
        graph.UpdateConsumerNodes(prev_input_node_ptr->Name(), new_consumer_nodes);
        // todo: need update the consumer node for the input_node as well.
      }
    }

    const std::vector<NodeArg*> sa_g_input_defs{dense_matmul_node.MutableOutputDefs()[0]};
    auto sa_g_type_info = *dense_matmul_node.MutableOutputDefs()[0]->TypeAsProto();  // copy
    //LOGS_DEFAULT(WARNING) << "%%%%%%%%%%%%%%%%%%%%%" << node.Name() << "dim 0: "
    //                      << sa_g_type_info.tensor_type().shape().dim(0).dim_value() << ", dim 1:"
    //                      << sa_g_type_info.tensor_type().shape().dim(1).dim_value();
    auto& sa_g_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("BARTAttention_MegatronG_Output"), &sa_g_type_info);
    Node& sa_g_node = graph.AddNode(graph.GenerateNodeName(k_matmul->Name() + "BARTAttention_MegatronG"),
                                    "MegatronG",
                                    "BARTAttention MegatronG",
                                    sa_g_input_defs,
                                    {&sa_g_out_arg}, {}, kMSDomain);
    sa_g_node.AddAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel));
    sa_g_node.SetExecutionProviderType(k_matmul->GetExecutionProviderType());
    graph_utils::ReplaceDownstreamNodeInput(graph, dense_matmul_node, 0, sa_g_node, 0);

    modified = true;
  }

  return Status::OK();
}

Status MegatronTransformer::TransformDropout(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger,
                                             std::unordered_set<Node*>& self_attention_dropout_nodes, int32_t& counter) const {
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
    if (self_attention_dropout_nodes.find(&node) != self_attention_dropout_nodes.end()) {
      int64_t seed = static_cast<int64_t>(HashName(node.MutableOutputDefs()[0]->Name())) + utils::GetRandomSeed();
      if (self_attention_dropout_nodes.find(&node) != self_attention_dropout_nodes.end()) {
        seed += horizontal_parallel_rank_;
      }

      if (graph_utils::GetNodeAttribute(node, "seed") != nullptr) {
        node.ClearAttribute("seed");
      }
      //std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< add seed for node " << node.Name() << std::endl;
      node.AddAttribute("seed", seed);
      counter++;
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

  int32_t partitioned_t5_mlp_count_ = 0;
  int32_t partitioned_t5_attention_count_ = 0;
  int32_t dropout_changed_ = 0;

  ORT_ENFORCE(TransformMLP(graph, modified, graph_level, logger, nodes_to_clear_shape).IsOK());
  ORT_ENFORCE(TransformT5MLP(graph, modified, graph_level, logger, nodes_to_clear_shape, self_attention_dropout_nodes, partitioned_t5_mlp_count_).IsOK());
  ORT_ENFORCE(TransformBARTMLP(graph, modified, graph_level, logger, nodes_to_clear_shape, self_attention_dropout_nodes, partitioned_t5_mlp_count_).IsOK());
  ORT_ENFORCE(TransformSelfAttention(graph, modified, graph_level, logger, nodes_to_clear_shape, self_attention_dropout_nodes).IsOK());
  ORT_ENFORCE(TransformT5SelfAttention(graph, modified, graph_level, logger, nodes_to_clear_shape, self_attention_dropout_nodes, partitioned_t5_attention_count_).IsOK());
  ORT_ENFORCE(TransformBARTSelfAttention(graph, modified, graph_level, logger, nodes_to_clear_shape, self_attention_dropout_nodes, partitioned_t5_attention_count_).IsOK());
  ORT_ENFORCE(TransformDropout(graph, modified, graph_level, logger, self_attention_dropout_nodes, dropout_changed_).IsOK());

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
    std::cout << "Megatron transformer result : Partitioned " << partitioned_t5_mlp_count_ << " T5 MLP Blocks, "
              << partitioned_t5_attention_count_ << " T5 Attention Blocks; Reset seed for " << dropout_changed_
              << " Dropout nodes. Error Message: " << ret.ErrorMessage() << std::endl;
    ORT_ENFORCE(ret.IsOK());
  }

  return Status::OK();
}

}  // namespace onnxruntime