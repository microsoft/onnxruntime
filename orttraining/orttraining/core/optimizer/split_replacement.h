// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class SplitReplacement

Rewrite rule that replaces Split with SplitView if the following conditions are satisfied:
- Split on axis 0
- Split has no any graph output
*/
class SplitReplacement : public RewriteRule {
 public:
  SplitReplacement() noexcept : RewriteRule("SplitReplacement") {}
  std::vector<std::string> TargetOpTypes() const noexcept override { return {"Split"}; }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;
  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
