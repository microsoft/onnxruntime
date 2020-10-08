// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class CommonSubexpressionElimination
Merge nodes that always evaluate to the same result.
*/
class CommonSubexpressionElimination : public GraphTransformer {
 public:
  CommonSubexpressionElimination(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("CommonSubexpressionElimination", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
