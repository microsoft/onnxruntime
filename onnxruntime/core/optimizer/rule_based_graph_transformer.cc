// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/rewrite_rule.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

Status RuleBasedGraphTransformer::Register(std::unique_ptr<RewriteRule> rule) {
  auto op_types = rule->TargetOpTypes();
  // If the target op types are empty, this rule will be evaluated for all op types.
  if (op_types.empty()) {
    any_op_type_rules_.push_back(std::move(rule));
  } else {
    std::for_each(op_types.cbegin(), op_types.cend(),
                  [&](const auto& op_type) { op_type_to_rules_[op_type].push_back(std::move(rule)); });
  }
  return Status::OK();
}

Status RuleBasedGraphTransformer::ApplyRulesOnNode(Graph& graph, Node& node,
                                                   const std::vector<std::unique_ptr<RewriteRule>>& rules,
                                                   RuleEffect& rule_effect) const {
  for (const auto& rule : rules) {
    ORT_RETURN_IF_ERROR(rule->CheckConditionAndApply(graph, node, rule_effect));
    // If the current node was removed as a result of a rule, stop rule application for that node.
    if (rule_effect == RuleEffect::kRemovedCurrentNode) {
      break;
    }
  }
  return Status::OK();
}

Status RuleBasedGraphTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    // A node might not be found as it might have already been deleted from one of the rules.
    if (!node) {
      continue;
    }

    // Initialize the effect of rules on this node to denote that the graph has not yet been modified
    // by the rule application on the current node.
    auto rule_effect = RuleEffect::kNone;
 
    if (!graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // First apply rewrite rules that are registered for the op type of the current node; then apply rules that are
    // registered to be applied regardless of the op type; then recursively apply rules to subgraphs (if any).
    // Stop further rule application for the current node, if the node gets removed by a rule.
    const std::vector<std::unique_ptr<RewriteRule>>* rules = nullptr;

    rules = GetRewriteRulesForOpType(node->OpType());
    if (rules) {
      ORT_RETURN_IF_ERROR(ApplyRulesOnNode(graph, *node, *rules, rule_effect));
    }

    if (rule_effect != RuleEffect::kRemovedCurrentNode) {
      rules = GetAnyOpRewriteRules();
      if (rules) {
        ORT_RETURN_IF_ERROR(ApplyRulesOnNode(graph, *node, *rules, rule_effect));
      }
    }

    // Update the modified field of the rule-based transformer.
    if (rule_effect != RuleEffect::kNone) {
      modified = true;
    }

    if (rule_effect != RuleEffect::kRemovedCurrentNode) {
      ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level));
    }
  }

  return Status::OK();
}

size_t RuleBasedGraphTransformer::RulesCount() const {
  return any_op_type_rules_.size() +
         std::accumulate(op_type_to_rules_.cbegin(), op_type_to_rules_.cend(), size_t(0),
                         [](size_t sum, const auto& rules) { return sum + rules.second.size(); });
}

}  // namespace onnxruntime
