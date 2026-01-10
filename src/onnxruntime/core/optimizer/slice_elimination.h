// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class EliminateSlice

Rewrite rule that eliminates a slice operator if it is redundant (does not lead to data reduction).

It is attempted to be triggered only on nodes with op type "Slice".
*/
class EliminateSlice : public RewriteRule {
 public:
  EliminateSlice() noexcept : RewriteRule("EliminateSlice") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Slice"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
