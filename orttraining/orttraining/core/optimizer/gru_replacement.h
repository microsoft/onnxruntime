// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GRUReplacement

This transformer is used when an GRU model is used for training. Training requires
extra set of outputs from the GRU cell that it needs to use during gradient computation.
So, this transformer will replace the existing GRU cell with the GRUTraining cell.

The extra output that the GRUTraining generates are:
ZRH intermediate gate computations needed for gradient computation.

*/

class GRUReplacement : public RewriteRule {
 public:
  GRUReplacement() noexcept : RewriteRule("GRUReplacement") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"GRU"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
