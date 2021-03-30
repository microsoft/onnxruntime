// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {
/**
@Class NotWhereFusion

Transform that fuses two Not -> Where nodes to a single Where node
with the where inputs 1 and 2 flipped.
Condition ->  Not -> Where ->
              value0-|  |
              value1----|

Condition -> Where ->
      value1-|  |
      value0----|
*/
class NotWhereFusion : public GraphTransformer {
 public:
  NotWhereFusion(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("NotWhereFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
