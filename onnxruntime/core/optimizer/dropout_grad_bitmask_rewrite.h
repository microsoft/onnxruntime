// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class DropoutGradGradBitmaskRewrite

Rewrite rule that converts a Dropout + DropoutGrad pairing into a BitmaskDropout + BitmaskDropoutGrad pairing.

This rule can only be applied when there are no other uses of the Dropout mask output. BitmaskDroput has a
different (and more memory-efficient) representation of the mask, which only BitmaskDropoutGrad is able to interpret
properly.

This is supported for Dropout opset >= 11, as this is the version that BitmaskDropout is designed to match up with.
 
This patterm is attempted to be triggered only on nodes with op type "Dropout".

         mask
Dropout -----> DropoutGrad

can be converted to

                  mask
BitmaskDropout --------> BitmaskDropoutGrad

All other inputs and outputs of these ops (such as the actual data output) are untouched, as BitmaskDropout{,Grad}
supports the same interface for the non-mask tensors.
*/
class DropoutGradBitmaskRewrite : public RewriteRule {
 public:
  DropoutGradBitmaskRewrite() noexcept : RewriteRule("DropoutGradBitmaskRewrite") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Dropout"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
