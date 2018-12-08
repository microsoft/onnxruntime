// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/rewrite_rule.h"

namespace onnxruntime {

/**
@class GraConstantFoldingphViewer

Rewrite rule that gets applied to nodes that have only initializers as inputs.
It computes these nodes and replaces their output with an initializer that corresponds
to the result of the computation.
*/
class ConstantFolding : public RewriteRule {
 public:
  ConstantFolding() noexcept : RewriteRule("ConstantFolding", "Constant folding") {}

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node) override;

  Status Apply(Graph& graph, Node& node, bool& modified) override;
};

}  // namespace onnxruntime
