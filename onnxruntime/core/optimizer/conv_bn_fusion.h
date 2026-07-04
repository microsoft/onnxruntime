// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class ConvBNFusion

Rewrite rule that fuses Conv/ConvTranspose+BN nodes to a single Conv/ConvTranspose node.

It is attempted to be triggered only on nodes with op type "Conv" or "ConvTranspose".
*/
class ConvBNFusion : public RewriteRule {
 public:
  ConvBNFusion() noexcept : RewriteRule("ConvBNFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Conv", "ConvTranspose"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
