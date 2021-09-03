// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class ShapeOptimization

Rewrite rule that removes redundant compute preceding a Shape op.
For example, :
1)
Cast -> Shape
Here, if the Cast has no other consumers, it is Redundant.

2)
Transpose -> Shape
If the Transpose has no other consumers, we can transpose the shape instead of 
transposing the larger-sized activation.
Shape -> Transpose

Similar patterns can be added to this Rewrite Rule as required.
*/
class ShapeOptimization : public RewriteRule {
 public:
  ShapeOptimization() noexcept : RewriteRule("ShapeOptimization") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Shape"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
