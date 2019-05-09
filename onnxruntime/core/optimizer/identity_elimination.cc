// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/op.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/identity_elimination.h"

namespace onnxruntime {

Status EliminateIdentity::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) {
  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool EliminateIdentity::SatisfyCondition(const Graph& graph, const Node& node) {
  return graph_utils::IsSingleInSingleOutNode(node) &&
         !graph.IsNodeOutputsInGraphOutputs(node);
}

}  // namespace onnxruntime
