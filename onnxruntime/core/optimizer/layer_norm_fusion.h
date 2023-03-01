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
  LayerNormFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
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
  SimplifiedLayerNormFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                            bool is_for_pre_training = false) noexcept
      : GraphTransformer("SimplifiedLayerNormFusion", compatible_execution_providers),
        is_for_pre_training_(is_for_pre_training) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  // A flag indicate whether this optimizer is registered for pre-training.
  // This is introduced to skip some device check, since when optimization passes are running, devices placement is
  // done yet.
  bool is_for_pre_training_;
};

}  // namespace onnxruntime
