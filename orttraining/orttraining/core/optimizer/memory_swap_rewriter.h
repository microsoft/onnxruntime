// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class MemorySwapRewriter

Rewrite rule for adding memory swap nodes.
*/
class MemorySwapRewriter : public RewriteRule {
 public:
  MemorySwapRewriter(const std::string& stop_at_node_arg) noexcept
      : RewriteRule("MemorySwap"),
        stop_at_node_arg_(stop_at_node_arg),
        stop_at_topo_index_(-1),
        last_graph_(nullptr),
        need_topo_sort_(false) {
  }

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {};  // enable for all nodes
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;

  mutable std::string stop_at_node_arg_;  // name of output[0] for the node to stop rewriter
  mutable int stop_at_topo_index_;
  mutable const Graph* last_graph_;
  mutable bool need_topo_sort_;
  mutable std::unordered_map<NodeIndex, int> topo_indices_;
};

}  // namespace onnxruntime
