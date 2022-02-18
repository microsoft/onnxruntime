// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class DropoutBitmaskRewrite

Rewrite rule that converts a Dropout op to a BitmaskDropout op, when certain invariants are met that make
such a transformation possible.

In particular, this transformation is only possible when there are no uses of the Dropout mask output. Only
DropoutGrad is aware of this different representation, and this is handled by the DropoutBitmaskGradRewrite pattern.

         mask
Dropout -----> UNUSED

can be rewritten to

                 mask
BitmaskDropout -------> UNUSED
*/
class DropoutBitmaskRewrite : public RewriteRule {
 public:
  DropoutBitmaskRewrite() noexcept : RewriteRule("DropoutBitmaskRewrite") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Dropout"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
