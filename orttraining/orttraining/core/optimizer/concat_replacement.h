// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class ConcatReplacement

Rewrite rule that replaces Concat with ConcatTraining, that has an additional output 
used in building the gradient for Concat node.

It is attempted to be triggered only on nodes with op type "Concat".
*/
class ConcatReplacement : public RewriteRule {
 public:
  ConcatReplacement() noexcept : RewriteRule("ConcatReplacement") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Concat"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
