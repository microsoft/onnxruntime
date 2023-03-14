// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class LSTMReplacement

This transformer is used when an LSTM model is used for training. Training requires
extra set of outputs from the LSTM cell that it needs to use during gradient computation.
So, this transformer will replace the existing LSTM cell with the LSTMTraining cell.

The extra set of arguments that the LSTMTraining generates are:
1). Cell states over all the sequence steps needed for gradient computation.
2). IOFC intermediate gate computations needed for gradient computation.

*/

class LSTMReplacement : public RewriteRule {
 public:
  LSTMReplacement() noexcept : RewriteRule("LSTMReplacement") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"LSTM"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
