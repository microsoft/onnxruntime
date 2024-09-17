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
 private:
  TransformerLevel optimization_level_ = TransformerLevel::Level1;
  bool allow_contrib_op_in_level_1_ = false;
  std::string GetLayerNormFusionName(TransformerLevel level) {
    switch (level) {
      case TransformerLevel::Level1:
        return "LayerNormFusionL1";
      case TransformerLevel::Level2:
        return "LayerNormFusionL2";
      default:
        return "LayerNormFusion";
    }
  }

 public:
  LayerNormFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                  TransformerLevel level = TransformerLevel::Level1, bool allow_contrib_op_in_level_1 = false) noexcept
      : GraphTransformer(GetLayerNormFusionName(level), compatible_execution_providers),
        optimization_level_(level),
        allow_contrib_op_in_level_1_(allow_contrib_op_in_level_1) {}

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
                            bool skip_device_check = false) noexcept
      : GraphTransformer("SimplifiedLayerNormFusion", compatible_execution_providers),
        skip_device_check_(skip_device_check) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  // A flag indicate whether device check is skipped for some cases.
  // This is introduced for pre-training optimizations, where when optimization passes are running,
  // devices placement is NOT done yet.
  bool skip_device_check_;
};

}  // namespace onnxruntime
