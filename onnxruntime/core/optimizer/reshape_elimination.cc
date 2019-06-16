// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/reshape_elimination.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/op.h"

namespace onnxruntime {

Status ReshapeElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) const {
  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool ReshapeElimination::SatisfyCondition(const Graph& graph, const Node& node) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Reshape", {1, 5})) {
    return false;
  }

  if (node.GetInputEdgesCount() != 1 || node.GetOutputEdgesCount() != 1) {
    return false;
  }

  auto next_node_itr = node.OutputNodesBegin();
  const Node& next_node = (*next_node_itr);
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Reshape", {1, 5})) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
