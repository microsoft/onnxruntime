// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class UnsqueezeElimination

Rewrite rule that eliminates an unsqueeze operator that takes as input an initializer.

It is attempted to be triggered only on nodes with op type "Unsqueeze".
*/
class UnsqueezeElimination : public RewriteRule {
 public:
  UnsqueezeElimination() noexcept : RewriteRule("UnsqueezeElimination") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Unsqueeze"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node) override;

  Status Apply(Graph& graph, Node& node, bool& modified, bool& deleted) override;
};

}  // namespace onnxruntime
