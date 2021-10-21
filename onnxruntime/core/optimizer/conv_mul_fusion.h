// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class ConvMulFusion

Rewrite rule that fuses two Conv+Mul nodes to a single Conv node.

It is attempted to be triggered only on nodes with op type "Conv".
*/
class ConvMulFusion : public RewriteRule {
 public:
  ConvMulFusion() noexcept : RewriteRule("ConvMulFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Conv"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
