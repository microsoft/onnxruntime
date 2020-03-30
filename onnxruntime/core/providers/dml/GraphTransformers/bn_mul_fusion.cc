// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#undef OPTIONAL
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "bn_mul_fusion.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status BatchNormalizationMulFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const onnxruntime::logging::Logger&) const {
  auto& BatchNormalization_node = node;
  const auto& mul_node = *BatchNormalization_node.OutputNodesBegin();
  const auto& BatchNormalization_inputs = BatchNormalization_node.InputDefs();
  const auto& mul_inputs = mul_node.InputDefs();

  const ONNX_NAMESPACE::TensorProto* BatchNormalization_Scale_tensor_proto = nullptr;
  if (!graph.GetInitializedTensor(BatchNormalization_inputs[1]->Name(), BatchNormalization_Scale_tensor_proto)) {
    return Status::OK();
  }

  const ONNX_NAMESPACE::TensorProto* mul_B_tensor_proto = nullptr;
  if (!graph.GetInitializedTensor(mul_inputs[1]->Name(), mul_B_tensor_proto)) {
    return Status::OK();
  }

  if (!optimizer_utils::IsFloatingPointDataType(*BatchNormalization_Scale_tensor_proto) ||
      !optimizer_utils::IsFloatingPointDataType(*mul_B_tensor_proto) ||
      BatchNormalization_Scale_tensor_proto->data_type() != mul_B_tensor_proto->data_type() ||
      BatchNormalization_Scale_tensor_proto->dims_size() != 1) {
    return Status::OK();
  }

  if (mul_B_tensor_proto->dims_size() != 0) {
    int axis;
    if (mul_B_tensor_proto->dims_size() >= 3 && mul_B_tensor_proto->dims_size() <= 4) {
      axis = mul_B_tensor_proto->dims_size() - 3;
    } else {
      return Status::OK();
    }
    if (mul_B_tensor_proto->dims(axis) != BatchNormalization_Scale_tensor_proto->dims(0)) {
      return Status::OK();
    }
    // The dimensions of mul_B should be equal to 1 except axis dimension.
    for (int i = 0; i < mul_B_tensor_proto->dims_size(); i++) {
      if (i != axis && mul_B_tensor_proto->dims(i) != 1) {
        return Status::OK();
      }
    }
  }

  Initializer BatchNormalization_Scale{*BatchNormalization_Scale_tensor_proto, graph.ModelPath()};
  Initializer mul_B{*mul_B_tensor_proto, graph.ModelPath()};

  const ONNX_NAMESPACE::TensorProto* BatchNormalization_B_tensor_proto = nullptr;
  if (!graph.GetInitializedTensor(BatchNormalization_inputs[2]->Name(), BatchNormalization_B_tensor_proto))
    return Status::OK();
  if (BatchNormalization_B_tensor_proto == nullptr)
    return Status(ONNXRUNTIME, FAIL, "Internal error in BatchNormalizationMulFusion. BatchNormalization_B_tensor_proto is NULL");
  if (!optimizer_utils::IsFloatingPointDataType(*BatchNormalization_B_tensor_proto) ||
      BatchNormalization_B_tensor_proto->data_type() != mul_B_tensor_proto->data_type() ||
      BatchNormalization_B_tensor_proto->dims_size() != 1) {
    return Status::OK();
  }
  Initializer BatchNormalization_B{*BatchNormalization_B_tensor_proto, graph.ModelPath()};

  // Calculate new value of initializers of BatchNormalization node
  BatchNormalization_Scale.scale_by_axis(mul_B, 1);

  if (mul_B_tensor_proto->dims_size() != 0) {
    BatchNormalization_B.mul(mul_B);
  } else {
    BatchNormalization_B.scale_by_axis(mul_B, 0);
  }

  // Create new initializers of BatchNormalization
  ONNX_NAMESPACE::TensorProto new_BatchNormalization_Scale_tensor_proto(*BatchNormalization_Scale_tensor_proto);
  BatchNormalization_Scale.ToProto(new_BatchNormalization_Scale_tensor_proto);

  // Replace initializers of BatchNormalization node
  graph.RemoveInitializedTensor(BatchNormalization_inputs[1]->Name());
  graph.AddInitializedTensor(new_BatchNormalization_Scale_tensor_proto);

  ONNX_NAMESPACE::TensorProto new_BatchNormalization_B_tensor_proto(*BatchNormalization_B_tensor_proto);
  BatchNormalization_B.ToProto(new_BatchNormalization_B_tensor_proto);
  graph.RemoveInitializedTensor(BatchNormalization_inputs[2]->Name());
  graph.AddInitializedTensor(new_BatchNormalization_B_tensor_proto);

  // Remove Mul node.
  auto* mul_node_to_remove = graph.GetNode(mul_node.Index());
  if (graph_utils::RemoveNode(graph, *mul_node_to_remove)) {
    rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  }

  return Status::OK();
}

bool BatchNormalizationMulFusion::SatisfyCondition(const Graph& graph, const Node& node, const onnxruntime::logging::Logger&) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "BatchNormalization", {7}) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto& next_node = *node.OutputNodesBegin();
  return !(!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Mul", {7}) ||
           next_node.GetInputEdgesCount() != 1 || !graph.GetNodeOutputsInGraphOutputs(next_node).empty() ||
           next_node.GetExecutionProviderType() != node.GetExecutionProviderType());
}

}  // namespace onnxruntime
