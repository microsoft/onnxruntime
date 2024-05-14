// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/transpose_replacement.h"

#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status TransposeReplacement::Apply(Graph& graph,
                                   Node& transpose_node,
                                   RewriteRuleEffect& rule_effect,
                                   const logging::Logger& logger) const {
  auto& transpose_inputs = transpose_node.MutableInputDefs();
  auto& transpose_outputs = transpose_node.MutableOutputDefs();
  NodeArg* input = transpose_inputs[0];
  auto input_shape = input->Shape();
  if (!input_shape) {
    LOG_DEBUG_INFO(logger, "Exit TransposeReplacement optimization for input shape is None.");
    return Status::OK();
  }
  auto perm = graph_utils::onnx_repeated_values::RetrieveValues<int64_t>(transpose_node.GetAttributes().at("perm"));
  InlinedVector<int64_t> new_shape;
  new_shape.reserve(perm.size());
  int64_t last_permuted_axis = 0;
  for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
    if (!input_shape->dim(static_cast<int>(perm[i])).has_dim_value()) {
      LOG_DEBUG_INFO(logger, "Exit TransposeReplacement optimization for not supporting symbolic shape.");
      return Status::OK();
    }
    new_shape.push_back(input_shape->dim(static_cast<int>(perm[i])).dim_value());
    if (input_shape->dim(static_cast<int>(perm[i])).dim_value() == 1)
      continue;
    if (perm[i] < last_permuted_axis) {
      LOG_DEBUG_INFO(logger, "Exit TransposeReplacement optimization for not supporting shape.");
      return Status::OK();
    }
    last_permuted_axis = perm[i];
  }

  transpose_inputs.push_back(
      optimizer::compute_optimizer::CreateInitializerFromVector(graph,
                                                                {static_cast<int64_t>(new_shape.size())},
                                                                new_shape,
                                                                graph.GenerateNodeArgName("transpose_reshape_shape")));

  Node& transpose_reshape_node = graph.AddNode(graph.GenerateNodeName("Transpose_Reshape"),
                                               "Reshape",
                                               "Transpose replaced Reshape",
                                               transpose_inputs,
                                               transpose_outputs,
                                               nullptr,
                                               kOnnxDomain);
  transpose_reshape_node.SetExecutionProviderType(transpose_node.GetExecutionProviderType());
  graph_utils::FinalizeNodeFusion(graph, transpose_reshape_node, transpose_node);
  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  return Status::OK();
}

bool TransposeReplacement::SatisfyCondition(const Graph&, const Node&, const logging::Logger&) const {
  return true;
}

}  // namespace onnxruntime
