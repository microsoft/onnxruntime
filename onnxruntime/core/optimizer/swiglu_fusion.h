// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * @brief Rewrite QuickGelu(gate) * up to com.microsoft SwiGLU(gate, up).
 *
 * QuickGeluFusion first turns gate * Sigmoid(alpha * gate) into QuickGelu, so running after it
 * covers the decomposed SiLU gated MLP that exporters emit today.
 */
class SwiGluFusion : public GraphTransformer {
 public:
  explicit SwiGluFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("SwiGluFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
