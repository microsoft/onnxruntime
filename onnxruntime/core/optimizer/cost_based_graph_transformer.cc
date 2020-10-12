// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/cost_based_graph_transformer.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/rewrite_rule.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

Status CostBasedGraphTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  // TODO: Use costs
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
    const std::vector<std::reference_wrapper<const RewriteRule>>* rules = nullptr;

    rules = GetRewriteRulesForOpType(node->OpType());
    if (rules) {
      ORT_RETURN_IF_ERROR(ApplyRulesOnNode(graph, *node, *rules, rule_effect, logger));
    }

    if (rule_effect != RuleEffect::kRemovedCurrentNode) {
      rules = GetAnyOpRewriteRules();
      if (rules) {
        ORT_RETURN_IF_ERROR(ApplyRulesOnNode(graph, *node, *rules, rule_effect, logger));
      }
    }

    // Update the modified field of the rule-based transformer.
    if (rule_effect != RuleEffect::kNone) {
      modified = true;
    }

    if (rule_effect != RuleEffect::kRemovedCurrentNode) {
      ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
