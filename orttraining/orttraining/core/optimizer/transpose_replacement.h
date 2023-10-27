// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/compute_optimizer/shared_utils.h"

namespace onnxruntime {

/**
@Class TransposeReplacement

Transpose is equivalent to a Reshape if:
 empty dimensions (which dim_value=1) can change place, not empty dimensions must be in
 the same order in the permuted tenosr.
 Example: Shape=(1,1,1024,4096) -> perm=(2,0,3,1).

This Rewrite rule replaces Transpose which meets the requirments with Reshape.
Because Transpose need memory copy while Reshape needn't, this replacement can save overhead for memory copy.

It is attempted to be triggered only on nodes with op type "Transpose".
*/
class TransposeReplacement : public RewriteRule {
 public:
  TransposeReplacement() noexcept : RewriteRule("TransposeReplacement") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Transpose"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
