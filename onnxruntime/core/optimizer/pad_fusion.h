// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {
/*
 *   This fusion submerges a Pad operator to it's child
 *   Conv or MaxPool operator, if and only if PadFusion::SatisfyCondition()
 *   is true.
 */
class PadFusion : public RewriteRule {
 public:
  PadFusion() : RewriteRule("Pad_Fusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Pad"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& matmul_node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};
}  // namespace onnxruntime