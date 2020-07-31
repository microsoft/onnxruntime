// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "orttraining/core/optimizer/concat_training_rewriter.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status ConcatTrainingRewriter::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  std::cout << "ConcatTrainingRewriter: Apply\n";
  auto& concat_node = node;
  const auto& concat_inputs = concat_node.MutableInputDefs();
  auto& concat_outputs = concat_node.MutableOutputDefs();

  ONNX_NAMESPACE::TypeProto t;
  t.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  t.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(concat_inputs.size());

  NodeArg& ip_shape_op = graph.GetOrCreateNodeArg("per_input_length", &t);

  concat_outputs.push_back(&ip_shape_op);

  Node& concat_training_node = graph.AddNode(graph.GenerateNodeName("ConcatTraining"),
                                             "ConcatTraining",
                                             "Concat with extra output",
                                             concat_inputs,
                                             concat_outputs,
                                             &concat_node.GetAttributes(),
                                             kMSDomain);

  // Assign provider to this new node. Provider should be same as the provider for old node.
  concat_training_node.SetExecutionProviderType(node.GetExecutionProviderType());
  graph_utils::FinalizeNodeFusion(graph, concat_training_node, concat_node);
  rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;
  return Status::OK();
}

bool ConcatTrainingRewriter::SatisfyCondition(const Graph&, const Node&, const logging::Logger&) const {
  std::cout << "ConcatTrainingRewriter: SatisfyCondition\n";

  return true;
}

}  // namespace onnxruntime
