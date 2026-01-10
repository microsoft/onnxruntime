// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
    @Class ClipQuantFusion

    Rewrite rule that fuses Clip into followed QuantizeLinear
 */
class ClipQuantFusion : public RewriteRule {
 public:
  ClipQuantFusion() noexcept : RewriteRule("ClipQuantRewrite") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Clip"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
