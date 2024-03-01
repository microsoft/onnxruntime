// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GemmTransposeB

Rewrite rule that transpose constant B if transB is true
*/
class GemmTransposeB : public RewriteRule {
 public:
  GemmTransposeB() noexcept : RewriteRule("GemmTransposeB") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Gemm"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
