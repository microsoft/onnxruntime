// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class LayerNormFusion

Rewrite graph fusing Layer Normalization subgraph to a single LayerNormalization node.

The formula corresponding to LayerNorm activation subgraph:
(x - mean(x, axis)) / sqrt(var(x, axis)) * scale + bias, where x is the input.

*/
class LayerNormFusion : public GraphTransformer {
 public:
  LayerNormFusion(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("LayerNormFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
