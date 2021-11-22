// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// gather_internal_replacement.j

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GatherInternalReplacement

Rewrite rule that replaces the GatherInternal with Gather when the extra
outputs computed by GatherInternal are not needed. This happens
when the GatherInternal node is not connected to a GatherGrad node.

*/
class GatherInternalReplacement : public RewriteRule {
 public:
  GatherInternalReplacement() noexcept : RewriteRule("GatherInternalReplacement") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"GatherInternal"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
