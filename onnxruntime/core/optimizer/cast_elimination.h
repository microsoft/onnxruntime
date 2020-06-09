// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class CastElimination

Rewrite rule that eliminates Cast nodes if 'to' attribute has same data type as input tensor data type.

It is attempted to be triggered only on nodes with op type "Cast".
*/
class CastElimination : public RewriteRule {
 public:
  CastElimination() noexcept : RewriteRule("CastElimination") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Cast"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
