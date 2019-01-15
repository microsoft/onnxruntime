// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_transformer.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

Status GraphTransformer::Apply(Graph& graph, bool& modified) const {
  ORT_RETURN_IF_ERROR(graph.Resolve());

  auto status = ApplyImpl(graph, modified, 0);
  ORT_RETURN_IF_ERROR(status);

  if (modified) {
    status = graph.Resolve();
  }

  return status;
}

Status RuleBasedGraphTransformer::Register(const std::string& op_type, std::unique_ptr<RewriteRule> rule) {
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
    if (rules) {
      for (const auto& rule : *rules) {
        ORT_RETURN_IF_ERROR(rule->CheckConditionAndApply(graph, *node, modified));
      }
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level));
  }

  return Status::OK();
}

}  // namespace onnxruntime
