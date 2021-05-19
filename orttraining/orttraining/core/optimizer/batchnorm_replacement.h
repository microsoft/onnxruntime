// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class BatchNorm Replacement

Rewrite rule that replaces BatchNorm with BatchNormInternal, that has additional outputs
for saved_mean and saved_std_dev
*/
class BatchNormReplacement : public RewriteRule {
 public:
  BatchNormReplacement() noexcept : RewriteRule("BatchNormReplacement") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"BatchNormalization"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
