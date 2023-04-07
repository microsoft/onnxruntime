// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GemmSumFusion

Rewrite rule that fuses Gemm and Sum nodes to a single Gemm node.
This fusion can be applied in the following scenario:
1) Sum at output of Gemm: when the output of a Gemm is immedietly summed with
    exactly one other element, we can fuse this Sum with Gemm by using the other
    Sum input as C, provided that the C input to the Gemm is missing.
    This is supported for opset >= 11, as this is when Gemm input C became optional.

TODO: Support the Add use case: Sum(x, y) ~= Add.
 
This patterm is attempted to be triggered only on nodes with op type "Gemm".

A --> Gemm --> D --> Sum --> E
       ^              ^
       |              |
B -----+              C

is equivalent to

A --> Gemm --> E
      ^  ^
      |  |
B ----+  C

Where each letter represents a tensor.
*/
class GemmSumFusion : public RewriteRule {
 public:
  GemmSumFusion() noexcept : RewriteRule("GemmSumFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Gemm"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
