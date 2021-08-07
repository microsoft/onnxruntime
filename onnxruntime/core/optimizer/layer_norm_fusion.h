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

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

/**
@Class SimplifiedLayerNormFusion

Rewrite graph fusing Layer Normalization subgraph to a single LayerNormalization node.

The formula corresponding to LayerNorm activation subgraph:
(x ) / sqrt(var(x, axis)) * scale, where x is the input, and var() is given by mean(x^2, axis).

*/
class SimplifiedLayerNormFusion : public GraphTransformer {
 public:
  SimplifiedLayerNormFusion(const std::unordered_set<std::string>& compatible_execution_providers = {},
                            const bool allow_precision_change = false) noexcept
      : GraphTransformer("SimplifiedLayerNormFusion", compatible_execution_providers),
        allow_precision_change_(allow_precision_change) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  bool allow_precision_change_;
};

}  // namespace onnxruntime
