// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class BatchNormalizationMulFusion

Rewrite rule that fuses two BatchNormalization+Mul nodes to a single BatchNormalization node.

It is attempted to be triggered only on nodes with op type "BatchNormalization".
*/
class BatchNormalizationMulFusion : public RewriteRule {
 public:
  BatchNormalizationMulFusion() noexcept : RewriteRule("BatchNormalizationMulFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"BatchNormalization"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const onnxruntime::logging::Logger&) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const onnxruntime::logging::Logger&) const override;
};

}  // namespace onnxruntime
