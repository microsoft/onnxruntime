// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class EliminateDropout

Rewrite rule that eliminates dropout nodes without downstream dependencies.

It is attempted to be triggered only on nodes with op type "Dropout".
*/
class EliminateDropout : public RewriteRule {
 public:
  EliminateDropout() noexcept : RewriteRule("EliminateDropout") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Dropout"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
