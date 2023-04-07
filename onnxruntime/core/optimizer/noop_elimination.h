// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class NoopElimination

Rewrite rule that eliminates the no op node.
Support x+0, 0+x, x-0, x*1, 1*x and x/1 for now.
*/
class NoopElimination : public RewriteRule {
 public:
  NoopElimination() noexcept : RewriteRule("NoopElimination") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Add", "Sub", "Mul", "Div"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};  // namespace onnxruntime

}  // namespace onnxruntime
