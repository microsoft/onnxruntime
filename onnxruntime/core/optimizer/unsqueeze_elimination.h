// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

class UnsqueezeElimination : public RewriteRule {
 public:
  UnsqueezeElimination() noexcept : RewriteRule("EliminateUnsqueeze", "Eliminate unsqueeze node") {}

 private:
  /** Apply rule when op type is one of the following. */
  const std::string included_op_type_ = "Unsqueeze";

  bool SatisfyCondition(const Graph& graph, const Node& node) override;

  bool OpTypeCondition(const Node& node) override;

  Status Apply(Graph& graph, Node& node, bool& modified, bool& deleted) override;
};

}  // namespace onnxruntime
