// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class EliminateIdentity

Rewrite rule that eliminates the identity node.

It is attempted to be triggered only on nodes with op type "Identity".
*/
class EliminateIdentity : public RewriteRule {
 public:
  EliminateIdentity() noexcept : RewriteRule("EliminateIdentity") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Identity"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};  // namespace onnxruntime

}  // namespace onnxruntime
