// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class ShapeToInitializer

When the input to a Shape operator is statically known (through shape inference), this rule replaces the Shape node
with an initializer to the downstream nodes.

It is attempted to be triggered only on nodes with op type "Shape".
*/
class ShapeToInitializer : public RewriteRule {
 public:
  ShapeToInitializer() noexcept : RewriteRule("ShapeToInitializer") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Shape"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
