// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/insert_output_rewriter.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

Status InsertMaxPoolOutput::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  auto& outputs = node.MutableOutputDefs();
  const NodeArg* Y = outputs[0];

  TypeProto t;
  t.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  if (Y->Shape() != nullptr) {
    t.mutable_tensor_type()->mutable_shape()->CopyFrom(*Y->Shape());
  }

  NodeArg& node_arg = graph.GetOrCreateNodeArg(Y->Name() + "_mask", &t);

  outputs.push_back(&node_arg);

  rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;
  return Status::OK();
}

bool InsertMaxPoolOutput::SatisfyCondition(const Graph& /*graph*/, const Node& node, const logging::Logger& /*logger*/) const {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {8, 10, 11, 12}) &&
      node.OutputDefs().size() == 1) {
    return true;
  }
  return false;
}

Status InsertSoftmaxCrossEntropyLossOutput::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  auto& outputs = node.MutableOutputDefs();
  const NodeArg* X = node.InputDefs()[0];

  TypeProto t;
  t.mutable_tensor_type()->set_elem_type(X->TypeAsProto()->tensor_type().elem_type());
  if (X->Shape() != nullptr) {
    t.mutable_tensor_type()->mutable_shape()->CopyFrom(*X->Shape());  // log probability should have the same shape as logits.
  }

  NodeArg& node_arg = graph.GetOrCreateNodeArg(X->Name() + "_log_prob", &t);

  outputs.push_back(&node_arg);

  rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;
  return Status::OK();
}

bool InsertSoftmaxCrossEntropyLossOutput::SatisfyCondition(const Graph& /*graph*/, const Node& node, const logging::Logger& /*logger*/) const {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "SoftmaxCrossEntropyLoss", {12}) &&
      node.OutputDefs().size() == 1) {
    return true;
  }
  return false;
}

Status AdjustBatchNormOutputs::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  auto& outputs = node.MutableOutputDefs();
  const auto& inputs = node.InputDefs();
  const NodeArg* scale_input_def = inputs[1];
  auto scale_input_def_type_proto = scale_input_def->TypeAsProto();

  NodeArg& running_mean_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("running_mean_def"), scale_input_def_type_proto);
  NodeArg& running_var_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("running_var_def"), scale_input_def_type_proto);
  NodeArg& saved_mean_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_mean_def"), scale_input_def_type_proto);
  NodeArg& saved_var_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_var_def"), scale_input_def_type_proto);

  outputs.push_back(&running_mean_def);
  outputs.push_back(&running_var_def);
  outputs.push_back(&saved_mean_def);
  outputs.push_back(&saved_var_def);

  // check Batch Normalization node has 5 output node args for training mode
  ORT_ENFORCE(node.OutputDefs().size() == 5);

  rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;
  return Status::OK();
}

bool AdjustBatchNormOutputs::SatisfyCondition(const Graph& /*graph*/, const Node& node, const logging::Logger& /*logger*/) const {
  if (node.OutputDefs().size() == 1) {
    return true;
  }
  return false;
}

}  // namespace onnxruntime
