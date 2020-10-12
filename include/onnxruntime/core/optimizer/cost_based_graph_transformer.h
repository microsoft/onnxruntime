// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/rule_based_graph_transformer.h"

namespace onnxruntime {

/**
@class CostBasedGraphTransformer

Cost-based graph transformer that provides an API to register rewrite rules
and an API to apply the optimal set of applicable rules to a Graph.

*/
class CostBasedGraphTransformer : public RuleBasedGraphTransformer {
 public:
  CostBasedGraphTransformer(const std::string& name,
                            const std::unordered_set<std::string>& compatible_execution_providers = {})
      : RuleBasedGraphTransformer(name, compatible_execution_providers) {}

 private:
  using RuleEffect = RewriteRule::RewriteRuleEffect;

  // The list of unique pointers for all rules (so that rules can be registered for several op types).
  std::vector<std::unique_ptr<RewriteRule>> rules_;
  // Map that associates a node's op type with the vector of rules that are registered to be triggered for that node.
  std::unordered_map<std::string, std::vector<std::reference_wrapper<const RewriteRule>>> op_type_to_rules_;
  // Rules that will be evaluated regardless of the op type of the node.
  std::vector<std::reference_wrapper<const RewriteRule>> any_op_type_rules_;
  // Performs a single top-down traversal of the graph and applies all registered rules.
  common::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
