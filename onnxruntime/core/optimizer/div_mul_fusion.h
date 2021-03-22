// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {
/**
@Class DivMulFusion

Transform that fuses two Div -> Mul nodes to a single Div node
when the first input to Div is 1.
1 / x1 *  x2 -> x2 / x1

*/
class DivMulFusion : public GraphTransformer {
 public:
  DivMulFusion(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("DivMulFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
