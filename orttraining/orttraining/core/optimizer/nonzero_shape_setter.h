// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

// Rewrite rule that set the output shape of NonZero Ops.
class NonZeroShapeSetter : public RewriteRule {
 public:
  NonZeroShapeSetter() noexcept
      : RewriteRule("NonZeroShapeSetter") {
  }

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"NonZero"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};
}  // namespace onnxruntime
