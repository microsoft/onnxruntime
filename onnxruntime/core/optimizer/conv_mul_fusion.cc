// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status ConvMulFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  auto& conv_node = node;
  auto& mul_node = *graph.GetNode(conv_node.OutputNodesBegin()->Index());

  const auto& conv_inputs = conv_node.InputDefs();
  const auto& mul_inputs = mul_node.InputDefs();

  const auto* conv_W_tensor_proto = graph_utils::GetConstantInitializer(graph, conv_inputs[1]->Name());
  ORT_ENFORCE(conv_W_tensor_proto);

  const auto* mul_B_tensor_proto = graph_utils::GetConstantInitializer(graph, mul_inputs[1]->Name());
  ORT_ENFORCE(mul_B_tensor_proto);

  // Conv only supports floating point data types, so can only fuse with an initializer containing those types
  if (!optimizer_utils::IsFloatingPointDataType(*conv_W_tensor_proto) ||
      conv_W_tensor_proto->data_type() != mul_B_tensor_proto->data_type() ||
      conv_W_tensor_proto->dims_size() <= 2) {
    return Status::OK();
  }

  if (mul_B_tensor_proto->dims_size() != 0) {
    int axis;
    if (mul_B_tensor_proto->dims_size() == conv_W_tensor_proto->dims_size()) {
      // Test for broadcast multiply such as 1xCx1x1 for a 2D convolution.
      axis = 1;
    } else if (mul_B_tensor_proto->dims_size() == conv_W_tensor_proto->dims_size() - 1) {
      // Test for broadcast multiply such as Cx1x1 for a 2D convolution.
      axis = 0;
    } else {
      return Status::OK();
    }
    if (mul_B_tensor_proto->dims(axis) != conv_W_tensor_proto->dims(0)) {
      return Status::OK();
    }
    // The dimensions of mul_B should be equal to 1 except axis dimension.
    for (int i = 0; i < mul_B_tensor_proto->dims_size(); i++) {
      if (i != axis && mul_B_tensor_proto->dims(i) != 1) {
        return Status::OK();
      }
    }
  }

  Initializer conv_W{*conv_W_tensor_proto, graph.ModelPath()};
  Initializer mul_B{*mul_B_tensor_proto, graph.ModelPath()};

  const ONNX_NAMESPACE::TensorProto* conv_B_tensor_proto = nullptr;
  std::unique_ptr<Initializer> conv_B = nullptr;
  const bool is_3d = conv_inputs.size() == 3;
  if (is_3d) {
    conv_B_tensor_proto = graph_utils::GetConstantInitializer(graph, conv_inputs[2]->Name());
    ORT_ENFORCE(conv_B_tensor_proto);

    if (conv_B_tensor_proto->data_type() != mul_B_tensor_proto->data_type() ||
        conv_B_tensor_proto->dims_size() != 1 ||
        conv_B_tensor_proto->dims(0) != conv_W_tensor_proto->dims(0)) {
      return Status::OK();
    }

    conv_B = onnxruntime::make_unique<Initializer>(*conv_B_tensor_proto, graph.ModelPath());
  }

  // Calculate new value of initializers of conv node
  conv_W.scale_by_axis(mul_B, 1);

  if (is_3d) {
    if (mul_B_tensor_proto->dims_size() != 0) {
      conv_B->mul(mul_B);
    } else {
      conv_B->scale_by_axis(mul_B, 0);
    }
  }

  // Create new initializers of conv
  ONNX_NAMESPACE::TensorProto new_conv_W_tensor_proto(*conv_W_tensor_proto);
  conv_W.ToProto(new_conv_W_tensor_proto);

  auto new_W_name = graph.GenerateNodeArgName("ConvMulFusion_W_" + conv_W_tensor_proto->name());
  new_conv_W_tensor_proto.set_name(new_W_name);

  // Replace initializers of conv node
  NodeArg& new_conv_W_node_arg = graph_utils::AddInitializer(graph, new_conv_W_tensor_proto);
  graph_utils::ReplaceNodeInput(conv_node, 1, new_conv_W_node_arg);

  if (is_3d) {
    ONNX_NAMESPACE::TensorProto new_conv_B_tensor_proto(*conv_B_tensor_proto);
    conv_B->ToProto(new_conv_B_tensor_proto);

    auto new_B_name = graph.GenerateNodeArgName("ConvMulFusion_Mul_B_" + mul_B_tensor_proto->name());
    new_conv_B_tensor_proto.set_name(new_B_name);

    NodeArg& new_conv_B_node_arg = graph_utils::AddInitializer(graph, new_conv_B_tensor_proto);
    graph_utils::ReplaceNodeInput(conv_node, 2, new_conv_B_node_arg);
  }

  // Move output name and edges from Mul node to Conv node and remove Mul node.
  graph_utils::FinalizeNodeFusion(graph, conv_node, mul_node);

  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}

bool ConvMulFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1, 11}) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Mul", {7, 13}) ||
      next_node.GetInputEdgesCount() != 1 ||
      // Make sure the two nodes do not span execution providers.
      next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  // Check that the appropriate inputs to the Conv and Mul nodels are constants.
  if (!graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[1]) ||
      (node.InputDefs().size() == 3 && !graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[2])) ||
      !graph_utils::NodeArgIsConstant(graph, *next_node.InputDefs()[1])) {
    return false;
  }

  if (!graph.GetNodeOutputsInGraphOutputs(node).empty()) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
