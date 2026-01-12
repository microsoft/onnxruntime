// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/cast_chain_elimination.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status CastChainElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  auto nextNodeIt = node.OutputNodesBegin();
  Node* next = graph.GetNode(nextNodeIt->Index());

  // We can remove the current node.
  graph_utils::RemoveNodeOutputEdges(graph, node);

  NodeArg* last_node_output_def = node.MutableOutputDefs()[0];
  const std::string& last_node_output_tensor_name = last_node_output_def->Name();

  // Find the matching def slot, so we can wire the final node to the input of the removeable node.
  int slot = -1;

  auto& inputs = next->MutableInputDefs();
  for (int i = 0, n = static_cast<int>(inputs.size()); i < n; ++i) {
    if (inputs[i]->Name() == last_node_output_tensor_name) {
      slot = i;
      break;
    }
  }

  next->MutableInputDefs()[slot] = node.MutableInputDefs()[0];

  graph_utils::MoveAllNodeInputEdges(graph, node, *next);

  graph.RemoveNode(node.Index());

  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

  return Status::OK();
}

bool CastChainElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (!graph_utils::CanRemoveNode(graph, node, logger)) {
    return false;
  }

  // Skip nodes that don't have 1 output edge.
  if (node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto nextNodeIt = node.OutputNodesBegin();

  const Node* next = graph.GetNode(nextNodeIt->Index());

  // Skip if the next node is not of type Cast.
  if (next->OpType() != "Cast") {
    return false;
  }

  return true;
}
}  // namespace onnxruntime
