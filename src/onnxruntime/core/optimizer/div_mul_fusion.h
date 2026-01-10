// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {
/**
@Class DivMulFusion

Rewrite rule that fuses two Div -> Mul nodes to a single Div node
when the first input to Div is 1.
1 / x1 *  x2 -> x2 / x1

*/
class DivMulFusion : public RewriteRule {
 public:
  DivMulFusion() noexcept : RewriteRule("DivMulFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Div"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
