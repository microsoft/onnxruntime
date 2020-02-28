// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#undef OPTIONAL
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "bn_add_fusion.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status BatchNormalizationAddFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& modified, const onnxruntime::logging::Logger&) const {
  auto& BatchNormalization_node = node;
  const auto& add_node = *BatchNormalization_node.OutputNodesBegin();
  const auto& BatchNormalization_inputs = BatchNormalization_node.InputDefs();
  const auto& add_inputs = add_node.InputDefs();

  const ONNX_NAMESPACE::TensorProto* BatchNormalization_Scale_tensor_proto = nullptr;
  if (!graph.GetInitializedTensor(BatchNormalization_inputs[1]->Name(), BatchNormalization_Scale_tensor_proto)) {
    return Status::OK();
  }

  const ONNX_NAMESPACE::TensorProto* add_B_tensor_proto = nullptr;
  if (!graph.GetInitializedTensor(add_inputs[1]->Name(), add_B_tensor_proto)) {
    return Status::OK();
  }

  // Currently, fusion is only supported for float or double data type.
  if (!optimizer_utils::IsFloatingPointDataType(*add_B_tensor_proto) ||
      BatchNormalization_Scale_tensor_proto->dims_size() != 1) {
    return Status::OK();
  }

  int axis;
  if (add_B_tensor_proto->dims_size() >= 3 && add_B_tensor_proto->dims_size() <= 4) {
    axis = add_B_tensor_proto->dims_size() - 3;
  } else {
    return Status::OK();
  }
  if (add_B_tensor_proto->dims(axis) != BatchNormalization_Scale_tensor_proto->dims(0)) {
    return Status::OK();
  }
  // The dimensions of add_B should be equal to 1 except axis dimension.
  for (int i = 0; i < add_B_tensor_proto->dims_size(); i++) {
    if (i != axis && add_B_tensor_proto->dims(i) != 1) {
      return Status::OK();
    }
  }

  const ONNX_NAMESPACE::TensorProto* BatchNormalization_B_tensor_proto = nullptr;
  if (!graph.GetInitializedTensor(BatchNormalization_inputs[2]->Name(), BatchNormalization_B_tensor_proto)) {
    return Status::OK();
  }

  if (!optimizer_utils::IsFloatingPointDataType(*BatchNormalization_B_tensor_proto) ||
      BatchNormalization_B_tensor_proto->data_type() != add_B_tensor_proto->data_type() ||
      BatchNormalization_B_tensor_proto->dims_size() != 1) {
    return Status::OK();
  }

  Initializer BatchNormalization_B{*BatchNormalization_B_tensor_proto, graph.ModelPath()};
  Initializer add_B{*add_B_tensor_proto, graph.ModelPath()};

  if (BatchNormalization_B.size() != add_B.size()) {
    return Status::OK();
  }
  // Calculate new value of initializers of BatchNormalization node
  BatchNormalization_B.add(add_B);

  // Create new initializers of BatchNormalization
  ONNX_NAMESPACE::TensorProto new_BatchNormalization_B_tensor_proto;
  BatchNormalization_B.ToProto(new_BatchNormalization_B_tensor_proto);

  // Replace initializers of BatchNormalization node
  graph.RemoveInitializedTensor(BatchNormalization_inputs[2]->Name());
  graph.AddInitializedTensor(new_BatchNormalization_B_tensor_proto);

  // Remove Add node.
  auto* add_node_to_remove = graph.GetNode(add_node.Index());
  if (graph_utils::RemoveNode(graph, *add_node_to_remove)) {
    modified = RewriteRuleEffect::kModifiedRestOfGraph;
  }

  return Status::OK();
}

bool BatchNormalizationAddFusion::SatisfyCondition(const Graph& graph, const Node& node, const onnxruntime::logging::Logger&) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "BatchNormalization", {7}) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto& next_node = *node.OutputNodesBegin();
  return !(!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Add", {7}) ||
           next_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
           next_node.GetInputEdgesCount() != 1 || !graph.GetNodeOutputsInGraphOutputs(next_node).empty());
}

}  // namespace onnxruntime
