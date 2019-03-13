// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

// Rewrite rule that eliminates the identity node.
class EliminateIdentity : public RewriteRule {
 public:
  EliminateIdentity() noexcept : RewriteRule("EliminateIdentity", "Eliminate identity node") {}

 private:
  /** Apply rule when op type is one of the following. */
  const std::string included_op_type_ = "Identity";

  bool SatisfyCondition(const Graph& graph, const Node& node) override;

  bool OpTypeCondition(const Node& node) override;

  Status Apply(Graph& graph, Node& node, bool& modified, bool& deleted) override;
};

}  // namespace onnxruntime
