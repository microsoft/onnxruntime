// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class GeluFusion

Rewrite graph fusing Gelu activation subgraph to a single Gelu node.

The formula corresponding to Gelu activation subgraph:
x * 0.5 * (1.0 + erf(x / sqrt(2.0))), where x is the input.

*/
class GeluFusion : public GraphTransformer {
 public:
  GeluFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("GeluFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
