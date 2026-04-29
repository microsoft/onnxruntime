// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * \class BiasSkipLayerNormFusion
 * \brief Rewrite graph fusing Add + SkipLayerNormalization subgraph to a single SkipLayerNormalization node,
 * where the Add node adds a 1D constant bias to the output of a MatMul (or Cast after MatMul).
 *
 * Before fusion:
 *     MatMul
 *       |
 *    Add(bias)    [skip]
 *            \       /
 *        SkipLayerNormalization (4 inputs: input, skip, gamma, beta)
 *
 * After fusion:
 *     MatMul    [skip]
 *         \       /
 *     SkipLayerNormalization (5 inputs: input, skip, gamma, beta, bias)
 */
class BiasSkipLayerNormFusion : public GraphTransformer {
 public:
  explicit BiasSkipLayerNormFusion(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("BiasSkipLayerNormFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
