// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {
/**
@Class LabelEncoderFusion

Rewrite rule that fuses two LabelEncoder -> LabelEncoder nodes to a single
LabelEncoder node.

*/
class LabelEncoderFusion : public RewriteRule {
 public:
  LabelEncoderFusion() noexcept : RewriteRule("LabelEncoderFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"LabelEncoder"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;
  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;

  template <class T1, class T2, class T3>
  Status ApplyHelper(Graph& graph, Node& node, Node& next_node, RewriteRuleEffect& rule_effect) const;
};

}  // namespace onnxruntime
