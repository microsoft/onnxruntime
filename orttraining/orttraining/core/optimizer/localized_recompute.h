// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GeluRecompute

Recompute Gelu/BiasGelu/FastGelu

*/
class GeluRecompute : public RewriteRule {
 public:
  GeluRecompute() noexcept : RewriteRule("GeluRecompute") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Gelu", "FastGelu", "BiasGelu"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

/**
@Class AttentionDropoutRecompute

Recompute Dropout in the attention layer

*/
class AttentionDropoutRecompute : public RewriteRule {
 public:
  AttentionDropoutRecompute() noexcept : RewriteRule("AttentionDropoutRecompute") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Dropout"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
