// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/conv_mul_fusion.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status ConvMulFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) {
  auto& conv_node = node;
  const auto& mul_node = *conv_node.OutputNodesBegin();
  const auto& conv_inputs = conv_node.InputDefs();
  const auto& mul_inputs = mul_node.InputDefs();

  const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
  graph.GetInitializedTensor(conv_inputs[1]->Name(), conv_W_tensor_proto);

  const ONNX_NAMESPACE::TensorProto* mul_B_tensor_proto = nullptr;
  graph.GetInitializedTensor(mul_inputs[1]->Name(), mul_B_tensor_proto);

  if (!Initializer::IsSupportedDataType(conv_W_tensor_proto) ||
      !Initializer::IsSupportedDataType(mul_B_tensor_proto) ||
      conv_W_tensor_proto->data_type() != mul_B_tensor_proto->data_type() ||
      conv_W_tensor_proto->dims_size() < 4 ||
      !(mul_B_tensor_proto->dims_size() == 0 ||
        (mul_B_tensor_proto->dims_size() == conv_W_tensor_proto->dims_size() - 1 &&
         conv_W_tensor_proto->dims(0) == mul_B_tensor_proto->dims(0)))) {
    return Status::OK();
  }

  // The dimensions of mul_B should be equal to 1 except first dimension.
  if (mul_B_tensor_proto->dims_size() != 0) {
    for (int i = 1; i < mul_B_tensor_proto->dims_size(); i++) {
      if (mul_B_tensor_proto->dims(i) != 1) {
        return Status::OK();
      }
    }
  }

  auto conv_W = std::make_unique<Initializer>(conv_W_tensor_proto);
  auto mul_B = std::make_unique<Initializer>(mul_B_tensor_proto);

  const ONNX_NAMESPACE::TensorProto* conv_B_tensor_proto = nullptr;
  std::unique_ptr<Initializer> conv_B = nullptr;
  const bool is_3d = conv_inputs.size() == 3;
  if (is_3d) {
    if (!graph.GetInitializedTensor(conv_inputs[2]->Name(), conv_B_tensor_proto))
      return Status::OK();
    if (conv_B_tensor_proto == nullptr)
      return Status(ONNXRUNTIME, FAIL, "Internal error in ConvMulFusion. conv_B_tensor_proto is NULL");
    if (!Initializer::IsSupportedDataType(conv_B_tensor_proto) ||
        conv_B_tensor_proto->data_type() != mul_B_tensor_proto->data_type() ||
        conv_B_tensor_proto->dims_size() != 1 ||
        (mul_B_tensor_proto->dims_size() != 0 && conv_B_tensor_proto->dims(0) != mul_B_tensor_proto->dims(0))) {
      return Status::OK();
    }
    conv_B = std::make_unique<Initializer>(conv_B_tensor_proto);
  }

  // Calculate new value of initializers of conv node
  conv_W->scale_by_axis(*mul_B, 1);

  if (conv_inputs.size() == 3) {
    if (mul_B_tensor_proto->dims_size() != 0) {
      conv_B->mul(*mul_B);
    } else {
      conv_B->scale_by_axis(*mul_B, 0);
    }
  }

  // Create new initializers of conv
  ONNX_NAMESPACE::TensorProto new_conv_W_tensor_proto(*conv_W_tensor_proto);
  conv_W->ToProto(&new_conv_W_tensor_proto);

  // Replace initializers of conv node
  graph.RemoveInitializedTensor(conv_inputs[1]->Name());
  graph.AddInitializedTensor(new_conv_W_tensor_proto);

  if (is_3d) {
    ONNX_NAMESPACE::TensorProto new_conv_B_tensor_proto(*conv_B_tensor_proto);
    conv_B->ToProto(&new_conv_B_tensor_proto);
    graph.RemoveInitializedTensor(conv_inputs[2]->Name());
    graph.AddInitializedTensor(new_conv_B_tensor_proto);
  }

  // Remove Mul node.
  auto* mul_node_to_remove = graph.GetNode(mul_node.Index());
  if (graph_utils::RemoveNode(graph, *mul_node_to_remove)) {
    rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  }

  return Status::OK();
}

bool ConvMulFusion::SatisfyCondition(const Graph& graph, const Node& node) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1}) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Mul", {7}) ||
      next_node.GetInputEdgesCount() != 1 ||
      graph.IsNodeOutputsInGraphOutputs(next_node) ||
      next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
