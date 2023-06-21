// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/pre_shape_node_elimination.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status PreShapeNodeElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  const Node* p_input_node = graph_utils::GetInputNode(node, 0);

  // Get mutable input node
  Node& input_node = *graph.GetNode(p_input_node->Index());
  const size_t output_idx = graph_utils::GetNodeOutputIndexFromOutputName(input_node, node.InputDefs()[0]->Name());

  std::vector<size_t> consumerIndices;

  // Save Consumer Nodes Indices (Shape)
  for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
    const auto& consumer = (*it).GetNode();
    const auto consumer_idx = consumer.Index();
    consumerIndices.push_back(consumer_idx);
  }

  // Remove output edges of the cast node
  graph_utils::RemoveNodeOutputEdges(graph, node);

  // Change input defs of shape nodes
  for (size_t consumer_idx : consumerIndices) {
    Node& shape_node = *graph.GetNode(consumer_idx);
    shape_node.MutableInputDefs()[0] = input_node.MutableOutputDefs()[output_idx];
  }

  graph.RemoveNode(node.Index());
  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

  return Status::OK();
}

bool PreShapeNodeElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Cast", {13, 15, 19}, kOnnxDomain)) {
    return false;
  }

  if (!graph_utils::CanRemoveNode(graph, node, logger)) {
    return false;
  }

  const Node* p_input_node = graph_utils::GetInputNode(node, 0);
  if (p_input_node == nullptr) {
    return false;
  }

  auto output_nodes = graph.GetConsumerNodes(node.OutputDefs()[0]->Name());

  if (output_nodes.empty()) {
    return false;
  }

  for (const Node* next_node : output_nodes) {
    // Check if the next node is not of type "Shape"
    if (next_node->OpType() != "Shape") {
      return false;
    }
  }

  // All output nodes are of type "Shape"
  return true;
}

}  // namespace onnxruntime
