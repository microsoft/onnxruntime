// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GemmTransposeFusion

Rewrite rule that fuses Gemm and Transpose nodes to a single Gemm node.
This fusion can be applied in the following scenarios:
1) Transpose at input(s) of Gemm: The Transpose can be removed and transA/B attr
    is set accordingly
2) Transpose at output of Gemm: The Transpose can be fused with Gemm by the rule:
    (AB)' = B' A'; provided that C input is missing.
    This is supported for Opset >= 11 as Gemm input C becomes optional from then
3) Transpose at Input(s) and Output: The fusion is applied by the rules in 1 and 2
 
It is attempted to be triggered only on nodes with op type "Gemm".
*/
class GemmTransposeFusion : public RewriteRule {
 public:
  GemmTransposeFusion() noexcept : RewriteRule("GemmTransposeFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Gemm"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
