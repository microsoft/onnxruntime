// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class ForceCpu

Rewrite rule that forces node to run on CPU EP.

This pattern is attempted to be triggered only on nodes with op type "Unsqueeze" and input type int64.
input --> Unsqueeze --> Unsqueeze --> Cast --> A
*/
class ForceCpu : public RewriteRule {
 public:
  ForceCpu() noexcept : RewriteRule("ForceCpu") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Unsqueeze"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
