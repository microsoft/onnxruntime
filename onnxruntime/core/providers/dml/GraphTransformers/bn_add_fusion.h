// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class BatchNormalizationAddFusion

Rewrite rule that fuses two BatchNormalization+Add nodes to a single BatchNormalization node.

It is attempted to be triggered only on nodes with op type "BatchNormalization".
*/
class BatchNormalizationAddFusion : public RewriteRule {
 public:
  BatchNormalizationAddFusion() noexcept : RewriteRule("BatchNormalizationAddFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"BatchNormalization"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const onnxruntime::logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const onnxruntime::logging::Logger& logger) const override;
};

}  // namespace onnxruntime
