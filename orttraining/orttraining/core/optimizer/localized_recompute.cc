// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "orttraining/core/graph/recompute_graph_utils.h"
#include "orttraining/core/optimizer/localized_recompute.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

bool GeluRecompute::SatisfyCondition(const Graph& /*graph*/, const Node& node, const logging::Logger& /*logger*/) const {
  const auto next_node = node.OutputNodesBegin();
  if (next_node != node.OutputNodesEnd() && next_node->OpType() == "MatMul") {
    return true;
  }
  return false;
}

Status GeluRecompute::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  const auto& output = node.OutputDefs()[0];

  auto& recomputed_output = graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                     output->TypeAsProto());

  graph.AddNode(node.Name() + "_recompute",
                node.OpType(),
                "Recompute of " + node.Name(),
                {node.MutableInputDefs()[0]},
                {&recomputed_output},
                &node.GetAttributes(),
                node.Domain());

  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  return Status::OK();
}

bool AttentionDropoutRecompute::SatisfyCondition(const Graph& /*graph*/, const Node& node, const logging::Logger& /*logger*/) const {
  const auto prev_node = node.InputNodesBegin();
  const auto next_node = node.OutputNodesBegin();
  if (prev_node != node.InputNodesEnd() && prev_node->OpType() == "Softmax" &&
      next_node != node.OutputNodesEnd() && next_node->OpType() == "MatMul") {
    return true;
  }
  return false;
}

Status AttentionDropoutRecompute::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  const auto& output = node.OutputDefs()[0];

  auto& recomputed_output = graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                     output->TypeAsProto());

  graph.AddNode(node.Name() + "_recompute",
                "DropoutGrad",                    // Reusing DropoutGrad as the recompute op 
                "Recompute of " + node.Name(),
                {
                    node.MutableInputDefs()[0],   // X
                    node.MutableOutputDefs()[1],  // mask
                    node.MutableInputDefs()[1],   // ratio
                    node.MutableInputDefs()[2]    // training_mode 
                },
                {&recomputed_output},
                {},
                kMSDomain);

  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  return Status::OK();
}

}  // namespace onnxruntime
