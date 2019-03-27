// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

// Rewrite rule that insert an addtional output to the matched node.
class InsertMaxPoolOutput : public RewriteRule {
 public:
  InsertMaxPoolOutput() noexcept
      : RewriteRule("InsertMaxPoolOutput", "Insert indices output to MaxPool") {
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node) override;

  Status Apply(Graph& graph, Node& node, bool& modified, bool& deleted) override;
};
}  // namespace onnxruntime
