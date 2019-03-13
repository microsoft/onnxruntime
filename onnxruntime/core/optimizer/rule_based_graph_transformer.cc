// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/rule_based_graph_transformer.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

Status RuleBasedGraphTransformer::Register(std::unique_ptr<RewriteRule> rule) {
  const auto& op_type = rule->OPType();
  if (HasRules(op_type)) {
    op_to_rules_[op_type] = std::vector<std::unique_ptr<RewriteRule>>();
  }

  op_to_rules_[op_type].push_back(std::move(rule));
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

    // Get the rules that should be fired for this node.
    const std::vector<std::unique_ptr<RewriteRule>>* rules = GetRewriteRules(node->OpType());

    bool deleted = false;
    if (rules) {
      for (const auto& rule : *rules) {
        ORT_RETURN_IF_ERROR(rule->CheckConditionAndApply(graph, *node, modified, deleted));
        if (deleted) {
          modified = true;  // should be set by rewriter but in case it wasn't...
          break;
        }
      }
    }

    if (!deleted) {
      ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level));
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
