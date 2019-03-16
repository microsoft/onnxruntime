// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

Status GraphTransformer::Apply(Graph& graph, bool& modified) const {
  // The Graph should be in a good state prior this being called, so there should be no need to call Resolve here.
  // ORT_RETURN_IF_ERROR(graph.Resolve());

  auto status = ApplyImpl(graph, modified, 0);
  ORT_RETURN_IF_ERROR(status);

  // At least currently, some transformers (InsertCastTransformer and MemcpyTransformer) need this to be called
  // after they complete to put the graph back into a valid state for the next transformer.
  if (modified) {
    status = graph.Resolve();
  }

  return status;
}

Status RuleBasedGraphTransformer::Register(std::unique_ptr<RewriteRule> rule) {
  rules_.push_back(std::move(rule));
  return Status::OK();
}

Status RuleBasedGraphTransformer::ApplyRulesOnNode(Graph& graph, Node& node,
                                                   const std::vector<std::unique_ptr<RewriteRule>>& rules,
                                                   bool& modified, bool& deleted) const {
  for (const auto& rule : rules) {
    ORT_RETURN_IF_ERROR(rule->CheckConditionAndApply(graph, node, modified, deleted));
    if (deleted) {
      modified = true;  // should be set by rewriter but in case it wasn't...
      break;
    }
  }
  return Status::OK();
}

Status TopDownRuleBasedTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    if (!node) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT);
    }

    // Apply rewrite rules on current node, then recursively apply rules to subgraphs (if any).
    // Stop further rule application for the current node, if the node gets removed by a rule.
    bool deleted = false;
    ORT_RETURN_IF_ERROR(ApplyRulesOnNode(graph, *node, GetRewriteRules(), modified, deleted));

    if (!deleted) {
      ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level));
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
