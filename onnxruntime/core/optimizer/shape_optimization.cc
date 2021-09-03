// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/shape_optimization.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/op.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

Status ShapeOptimization::Apply(Graph& graph, Node& shape_node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  Node& previous_node = *graph.GetNode(shape_node.InputNodesBegin()->Index());

  auto op_type = previous_node.OpType();
  if (op_type == "Cast") {
    Node& cast_node = previous_node;

    const Node* cast_input_node = graph_utils::GetInputNode(cast_node, 0);
    NodeArg* new_input_arg_for_shape = cast_node.MutableInputDefs()[0];

    int output_idx = -1;
    if (cast_input_node) {
      output_idx = graph_utils::GetNodeOutputIndexFromOutputName(*cast_input_node, new_input_arg_for_shape->Name());
      graph.RemoveEdge(cast_input_node->Index(), cast_node.Index(), output_idx, 0);
    }
    graph.RemoveEdge(cast_node.Index(), shape_node.Index(), 0, 0);
    graph_utils::ReplaceNodeInput(shape_node, 0, *new_input_arg_for_shape);

    if (cast_input_node) {
      graph.AddEdge(cast_input_node->Index(), shape_node.Index(), output_idx, 0);
    }
    // remove cast_node
    graph.RemoveNode(cast_node.Index());

    rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;

  } else if (op_type == "Transpose") {
#ifndef DISABLE_CONTRIB_OPS
    auto& transpose_node = previous_node;
    // add TransposeOfShape node
    const std::vector<NodeArg*> new_input_defs{transpose_node.MutableInputDefs()[0]};
    Node& fused_node = graph.AddNode(graph.GenerateNodeName("TransposeOfShape"),
                                     "TransposeOfShape",
                                     "fused transpose-shape subgraph ",
                                     new_input_defs,
                                     {}, {}, kMSDomain);
    auto attributes = transpose_node.GetAttributes();
    if (attributes.find("perm") != attributes.end()) {
      auto perm = ONNX_NAMESPACE::RetrieveValues<int64_t>(attributes.at("perm"));
      fused_node.AddAttribute("perm", perm);
    }

    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_node.SetExecutionProviderType(shape_node.GetExecutionProviderType());

    // move the output definition and edges from the mul_node to the div_node and delete the mul_node
    graph_utils::FinalizeNodeFusion(graph, {transpose_node, shape_node}, fused_node);

    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

#else
    rule_effect = RewriteRuleEffect::kNone;
#endif
  } else {
    rule_effect = RewriteRuleEffect::kNone;
  }

  return Status::OK();
}

bool ShapeOptimization::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  // TODO: Support Opset-15 Shape. There needs to be some additional consideration for
  // modifying the 'perm' with added attributes of shape 'start' and 'end'.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Shape", {1, 13})) {
    return false;
  }

  // There has to be an input edge as we are going to look at the previous node.
  if (node.GetInputEdgesCount() != 1) {
    return false;
  }

  // The previous nodes should be one of the implemented types,
  // and shouldn't have more than one Consumer.
  // In case of Transpose, where both the nodes are kept and just the node order is swapped,
  // the execution providers must match.
  const auto& previous_node = *node.InputNodesBegin();
  if (!(graph_utils::IsSupportedOptypeVersionAndDomain(previous_node, "Cast", {1, 6, 9, 13}) ||
        graph_utils::IsSupportedOptypeVersionAndDomain(previous_node, "Transpose", {1, 13})) ||
      previous_node.GetOutputEdgesCount() > 1 ||
      previous_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  if (!graph_utils::CanRemoveNode(graph, previous_node, logger)) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
