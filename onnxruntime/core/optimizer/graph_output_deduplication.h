// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// graph_output_deduplication.h

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GraphOutputDeduplication

Rewrite rule that deduplicates duplicate graph outputs by introducing a Identity op.

It is attempted to be triggered on all nodes.
*/
class GraphOutputDeduplication : public RewriteRule {
 public:
  GraphOutputDeduplication() noexcept : RewriteRule("GraphOutputDeduplication") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    // Trigger rule for all nodes.
    return {};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};  // namespace onnxruntime

}  // namespace onnxruntime
