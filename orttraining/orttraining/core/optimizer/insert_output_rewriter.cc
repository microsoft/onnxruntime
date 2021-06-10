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
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "SoftmaxCrossEntropyLoss", {12, 13}) &&
      node.OutputDefs().size() == 1) {
    return true;
  }
  return false;
}

}  // namespace onnxruntime
