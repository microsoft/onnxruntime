// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

// Rewrite rule that eliminates a slice operator if it is redundant (does not lead to data reduction).
class EliminateSlice : public RewriteRule {
 public:
  EliminateSlice() noexcept : RewriteRule("EliminateSlice", "Eliminate slice node") {}

 private:
  /** Apply rule when op type is one of the following. */
  const std::string included_op_type_ = "Slice";

  bool SatisfyCondition(const Graph& graph, const Node& node) override;

  bool OpTypeCondition(const Node& node) override;

  Status Apply(Graph& graph, Node& node, bool& modified, bool& removed) override;
};

}  // namespace onnxruntime
