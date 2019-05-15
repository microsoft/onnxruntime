// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class EliminateRelu

Rewrite rule that eliminates a Relu operator if it is redundant due to a following Clip operator.
*/
class EliminateRelu : public RewriteRule {
 public:
  EliminateRelu() noexcept : RewriteRule("EliminateRelu") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Relu"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node) override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) override;
};

}  // namespace onnxruntime
