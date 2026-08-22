// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/mul_add_fusion.h"
#include "core/optimizer/utils.h"
#include "core/common/logging/logging.h"
#include "core/framework/data_types.h"
#include "core/framework/tensorprotoutils.h"  // For utilities like TensorProtoToMLFloat16 etc.

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

bool IsPatternBatchnorm(const NodeArg* inp,
                        const ONNX_NAMESPACE::TensorProto* scale,
                        const ONNX_NAMESPACE::TensorProto* bias,
                        const logging::Logger& logger) {
  if (!inp->Shape()) {
    LOGS(logger, VERBOSE) << "Skip MulAddFusion since " << inp->Name() << " has nullptr on shape";
    return false;
  }
  int inp_rank = inp->Shape()->dim_size();
  int scale_rank = scale->dims_size();
  int bias_rank = bias->dims_size();
  int max_rank = std::max({inp_rank, scale_rank, bias_rank});
  if (max_rank <= 1) {
    LOGS(logger, VERBOSE) << "Skip MulAddFusion since the max rank among " << inp->Name() << ", " << scale->name() << " and " << bias->name() << "is <= 1";
    return false;
  }
  std::vector<int64_t> broadcast_inp(max_rank);
  std::vector<int64_t> broadcast_scale(max_rank);
  std::vector<int64_t> broadcast_bias(max_rank);
  for (int idx = 0; idx < max_rank; ++idx) {
    auto broad_idx = max_rank - 1 - idx;
    broadcast_inp[broad_idx] = (idx < inp_rank) ? inp->Shape()->dim(inp_rank - 1 - idx).dim_value() : 1;
    broadcast_scale[broad_idx] = (idx < scale_rank) ? scale->dims(scale_rank - 1 - idx) : 1;
    broadcast_bias[broad_idx] = (idx < bias_rank) ? bias->dims(bias_rank - 1 - idx) : 1;
  }
  // broadcast_scale and broadcast_bias should be in the form of [1, num_channel, 1, ..., 1].
  // Note: The num_channel can be 1
  int64_t num_channel = broadcast_inp[1];
  if ((broadcast_scale[0] != 1) || (broadcast_scale[1] != 1 && broadcast_scale[1] != num_channel)) {
    LOGS(logger, VERBOSE) << "Skip MulAddFusion since " << scale->name() << " has unsupported shape.";
    return false;
  }
  if ((broadcast_bias[0] != 1) || (broadcast_bias[1] != 1 && broadcast_bias[1] != num_channel)) {
    LOGS(logger, VERBOSE) << "Skip MulAddFusion since " << bias->name() << " has unsupported shape.";
    return false;
  }
  for (int idx = 2; idx < max_rank; ++idx) {
    if (broadcast_scale[idx] != 1 || broadcast_bias[idx] != 1) {
      LOGS(logger, VERBOSE) << "Skip MulAddFusion since " << scale->name() << " or " << bias->name() << " has unsupported shape.";
      return false;
    }
  }
  return true;
}

bool MulAddFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  auto& mul_node = node;
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7}) ||
      mul_node.GetOutputEdgesCount() != 1) {
    return false;
  }
  const auto& add_node = *mul_node.OutputNodesBegin();
  // Make sure the two nodes do not span execution providers.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7}) ||
      (add_node.GetExecutionProviderType() != mul_node.GetExecutionProviderType())) {
    return false;
  }
  // Pattern: Input -> Mul -> Add
  if (mul_node.InputDefs().size() != 2 || add_node.InputDefs().size() != 2) {
    LOGS(logger, VERBOSE) << "Skip MulAddFusion since " << mul_node.Name() << " or " << add_node.Name() << " has more than 2 inputs.";
    return false;
  }

  // Get the second input of Mul (scale) and Add (bias)
  // Focus on the case mul and add have exactly one constant and one non-constant input
  bool is_const_mul_in0 = graph_utils::NodeArgIsConstant(graph, *mul_node.InputDefs()[0]);
  bool is_const_mul_in1 = graph_utils::NodeArgIsConstant(graph, *mul_node.InputDefs()[1]);
  bool is_const_add_in0 = graph_utils::NodeArgIsConstant(graph, *add_node.InputDefs()[0]);
  bool is_const_add_in1 = graph_utils::NodeArgIsConstant(graph, *add_node.InputDefs()[1]);
  if ((is_const_mul_in0 && is_const_mul_in1) || (!is_const_mul_in0 && !is_const_mul_in1)) {
    LOGS(logger, VERBOSE) << "Skip MulAddFusion since " << mul_node.Name() << " should have exactly 1 constant and 1 non-contant input.";
    return false;
  }
  if ((is_const_add_in0 && is_const_add_in1) || (!is_const_add_in0 && !is_const_add_in1)) {
    LOGS(logger, VERBOSE) << "Skip MulAddFusion since " << add_node.Name() << " should have exactly 1 constant and 1 non-contant input.";
    return false;
  }

  auto mul_const_idx = is_const_mul_in0 ? 0 : 1;
  auto add_const_idx = is_const_add_in0 ? 0 : 1;
  return IsPatternBatchnorm(
      mul_node.InputDefs()[1 - mul_const_idx],
      graph_utils::GetConstantInitializer(graph, mul_node.InputDefs()[mul_const_idx]->Name()),
      graph_utils::GetConstantInitializer(graph, add_node.InputDefs()[add_const_idx]->Name()),
      logger);
}

Status MulAddFusion::FuseMulAdd(Node& node, Graph& graph, bool& modified, const logging::Logger&) const {
  auto& mul_node = node;
  Node& add_node = *graph.GetNode(mul_node.OutputNodesBegin()->Index());
  bool is_const_mul_in0 = graph_utils::NodeArgIsConstant(graph, *mul_node.InputDefs()[0]);
  bool is_const_add_in0 = graph_utils::NodeArgIsConstant(graph, *add_node.InputDefs()[0]);
  auto mul_const_idx = is_const_mul_in0 ? 0 : 1;
  auto mul_non_const_idx = 1 - mul_const_idx;
  auto add_const_idx = is_const_add_in0 ? 0 : 1;
  // Before layout transform, channel is the 1st dimension
  int64_t num_channel = mul_node.InputDefs()[mul_non_const_idx]->Shape()->dim(1).dim_value();

  // Process scale and bias. Should be {num_channel}
  const auto* scale_tensor_proto = graph_utils::GetConstantInitializer(graph, mul_node.InputDefs()[mul_const_idx]->Name());
  const auto* bias_tensor_proto = graph_utils::GetConstantInitializer(graph, add_node.InputDefs()[add_const_idx]->Name());
  ORT_ENFORCE(scale_tensor_proto);
  ORT_ENFORCE(bias_tensor_proto);
  ONNX_NAMESPACE::TensorProto reshaped_scale_proto = *scale_tensor_proto;
  ONNX_NAMESPACE::TensorProto reshaped_bias_tensor_proto = *bias_tensor_proto;
  reshaped_scale_proto.clear_dims();
  reshaped_scale_proto.set_name(scale_tensor_proto->name() + "_reshaped");
  reshaped_scale_proto.add_dims(num_channel);
  reshaped_bias_tensor_proto.clear_dims();
  reshaped_bias_tensor_proto.set_name(bias_tensor_proto->name() + "_reshaped");
  reshaped_bias_tensor_proto.add_dims(num_channel);
  NodeArg& reshaped_scale_node_arg = graph_utils::AddInitializer(graph, reshaped_scale_proto);
  NodeArg& reshaped_bias_node_arg = graph_utils::AddInitializer(graph, reshaped_bias_tensor_proto);

  // add initializer of mean as zeros of shape [channel]
  Initializer mean_init(
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(mul_node.InputDefs()[mul_non_const_idx]->TypeAsProto()->tensor_type().elem_type()),
      graph.GenerateNodeArgName(mul_node.Name() + "_mul_add_fusion_mean"),
      gsl::span<const int64_t>({num_channel}));
  ONNX_NAMESPACE::TensorProto mean_tensor_proto;
  mean_init.ToProto(mean_tensor_proto);
  NodeArg& mean_init_node_arg = graph_utils::AddInitializer(graph, mean_tensor_proto);

  // add initializer of var as ones of shape [channel]
  Initializer var_init(
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(mul_node.InputDefs()[mul_non_const_idx]->TypeAsProto()->tensor_type().elem_type()),
      graph.GenerateNodeArgName(add_node.Name() + "_mul_add_fusion_var"),
      gsl::span<const int64_t>({num_channel}));
  var_init.add(1);
  ONNX_NAMESPACE::TensorProto var_tensor_proto;
  var_init.ToProto(var_tensor_proto);
  NodeArg& var_init_node_arg = graph_utils::AddInitializer(graph, var_tensor_proto);

  // add BatchNormalization
  Node& bn_node = graph.AddNode(
      graph.GenerateNodeName(mul_node.Name() + "/MulAddFusion"),
      "BatchNormalization",
      "fused Mul and Add",
      gsl::span<NodeArg* const>({mul_node.MutableInputDefs()[mul_non_const_idx],
                                 &reshaped_scale_node_arg,
                                 &reshaped_bias_node_arg,
                                 &mean_init_node_arg,
                                 &var_init_node_arg}),
      gsl::span<NodeArg* const>({add_node.MutableOutputDefs()[0]}),
      nullptr,
      kOnnxDomainAlias);
  bn_node.SetExecutionProviderType(mul_node.GetExecutionProviderType());
  constexpr float eps = 0.0f;
  bn_node.SetSinceVersion(9);
  bn_node.AddAttribute("epsilon", eps);

  auto mul_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(mul_node);
  auto add_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(add_node);
  if (!graph_utils::IsGraphInput(graph, mul_node.InputDefs()[mul_non_const_idx])) {
    graph.AddEdge(
        mul_input_edges[mul_non_const_idx].src_node,
        bn_node.Index(),
        mul_input_edges[mul_non_const_idx].src_arg_index,
        0);
  }

  graph_utils::GraphEdge::RemoveGraphEdges(graph, mul_input_edges);
  graph_utils::GraphEdge::RemoveGraphEdges(graph, add_input_edges);
  graph_utils::RemoveNodeOutputEdges(graph, add_node);
  graph_utils::ReplaceDownstreamNodeInput(graph, add_node, 0, bn_node, 0);
  graph.RemoveNode(mul_node.Index());
  graph.RemoveNode(add_node.Index());

  modified = true;
  return Status::OK();
}

Status MulAddFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  const GraphViewer graph_viewer{graph};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    auto* node_ptr = graph.GetNode(node_idx);
    if (!node_ptr) {
      continue;
    }

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (this->SatisfyCondition(graph, node, logger)) {
      ORT_RETURN_IF_ERROR(this->FuseMulAdd(node, graph, modified, logger));
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
