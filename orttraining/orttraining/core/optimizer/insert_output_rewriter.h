// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

// Rewrite rule that insert an addtional output to the matched node.
class InsertMaxPoolOutput : public RewriteRule {
 public:
  InsertMaxPoolOutput() noexcept
      : RewriteRule("InsertMaxPoolOutput") {
  }

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

// Rewrite rule that insert an addtional output to the matched node.
// Adding this second output to expose FW intermediate result for speeding up BW computation 
class InsertSoftmaxCrossEntropyLossOutput : public RewriteRule {
 public:
  InsertSoftmaxCrossEntropyLossOutput() noexcept
      : RewriteRule("InsertSoftmaxCrossEntropyLossOutput") {
  }

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"SoftmaxCrossEntropyLoss"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
