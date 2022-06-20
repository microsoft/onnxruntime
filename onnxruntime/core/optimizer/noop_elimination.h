// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class NoopElimination

Rewrite rule that eliminates the no op node.
So far only Add node with 0 as one of its inputs is eliminated.
But this class could be the placeholder for other no op nodes in future.  
*/
class NoopElimination : public RewriteRule {
 public:
  NoopElimination() noexcept : RewriteRule("NoopElimination") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Add"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};  // namespace onnxruntime

}  // namespace onnxruntime
