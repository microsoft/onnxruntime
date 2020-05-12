// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class PromoteTopologicalOrderRewriter

Rewrite rule for add control edges to promote topological order of given nodes.
*/
class PromoteTopologicalOrderRewriter : public RewriteRule {
 public:
  PromoteTopologicalOrderRewriter() noexcept
      : RewriteRule("PromoteTopologicalOrderRewriter") {
  }

  std::vector<std::string> TargetOpTypes() const noexcept override;

 private:
  bool SatisfyCondition(const Graph& /*graph*/, const Node& node, const logging::Logger& /*logger*/) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;

  void AddEdgeInForward(Graph& graph, Node& node, const std::unordered_map<NodeIndex, int>& topo_indices) const;
};

}  // namespace onnxruntime
