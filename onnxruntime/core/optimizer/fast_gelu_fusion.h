// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

struct MatchResult {
  public:
    bool matched;
    NodeArg* gelu_without_bias_input_arg; // The Gelu input arg if not considering bias node.
    Node* tanh_input_node;
};

/**
@Class FastGeluFusion

Rewrite graph fusing Gelu activation subgraph to a single Gelu node.

The formula corresponding to Gelu activation subgraph:
x * 0.5 * (1.0 + tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x))) or
x * 0.5 * (1.0 + tanh((sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))), where x is the input.

*/
class FastGeluFusion : public GraphTransformer {
 public:
  FastGeluFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("FastGeluFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

private:
  MatchResult CheckFirstFormula(Graph& graph, Node& node, InlinedVector<std::reference_wrapper<Node>>& nodes_to_fuse) const;

  MatchResult CheckSecondFormula(Graph& graph, Node& nodes, InlinedVector<std::reference_wrapper<Node>>& nodes_to_fuse) const;
};

}  // namespace onnxruntime
