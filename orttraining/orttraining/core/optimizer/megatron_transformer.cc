// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "orttraining/core/optimizer/megatron_transformer.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

bool MegatronTransformer::PartitionWeightByColumn(const Graph& graph, const NodeArg& input_arg,
                                                  ONNX_NAMESPACE::TensorProto& initializer_partition) const {
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

  if (column_count % horizontal_parallel_size_ != 0) {
    LOGS_DEFAULT(WARNING) << "last dim " << column_count << " is not divisible by model_parallel_size "
                          << horizontal_parallel_size_ << ", not supported currently.";
    return false;
  }

  auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto);
  const float* a_weight = initializer->data<float>();

  initializer_partition.set_name("rank_" + std::to_string(horizontal_parallel_rank_) +
                                 "_" + input_arg.Name() + "_partition");
  initializer_partition.set_data_type(data_type);

  int64_t column_partition = column_count / horizontal_parallel_size_;

  if (rank == 2) {
    initializer_partition.add_dims(row_count);
  }

  initializer_partition.add_dims(column_partition);
  const int64_t element_count = row_count * column_partition;

  std::vector<float> result;
  result.reserve(element_count);

  const int64_t column_index_offset = horizontal_parallel_rank_ * column_partition;
  for (auto row_index = 0; row_index < row_count; row_index++) {
    for (auto column_index = 0; column_index < column_partition; column_index++) {
      result.push_back(a_weight[row_index * column_count + column_index_offset + column_index]);
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

  auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto);
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
    mlp_g_node.SetExecutionProviderType(node.GetExecutionProviderType());
    graph_utils::ReplaceDownstreamNodeInput(graph, matmul2_node, 0, mlp_g_node, 0);
    modified = true;
  }

  return Status::OK();
}

Status MegatronTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  if (horizontal_parallel_size_ <= 1) {
    return Status::OK();
  }

  std::vector<Node*> nodes_to_clear_shape;

  ORT_ENFORCE(TransformMLP(graph, modified, graph_level, logger, nodes_to_clear_shape).IsOK());

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
