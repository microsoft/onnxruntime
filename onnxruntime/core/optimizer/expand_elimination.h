// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class ExpandElimination

Rewrite rule that eliminates Expand nodes if the node generate the same tensor as the input tensor.

It is attempted to be triggered only on nodes with op type "Expand".
*/
class ExpandElimination : public RewriteRule {
 public:
  ExpandElimination() noexcept : RewriteRule("ExpandElimination") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Expand"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
