// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {
/**
@Class NotWhereFusion

Rewrite rule that fuses two Not -> Where nodes to a single Where node
with the where inputs 1 and 2 flipped.
Condition ->  Not -> Where ->
              value0-|  |
              value1----|

Condition -> Where ->
      value1-|  |
      value0----|
*/
class NotWhereFusion : public RewriteRule {
 public:
  NotWhereFusion() noexcept : RewriteRule("NotWhereFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Where"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
