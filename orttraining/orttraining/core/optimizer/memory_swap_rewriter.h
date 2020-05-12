// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class MemorySwapRewriter

Rewrite rule for adding memory swap nodes.
*/
class MemorySwapRewriter : public RewriteRule {
 public:
  MemorySwapRewriter(int min_topo_distance) noexcept
      : RewriteRule("MemorySwap"),
        min_topo_distance_(min_topo_distance) {
  }

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {};  // enable for all nodes
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;

  int min_topo_distance_;
};

}  // namespace onnxruntime
