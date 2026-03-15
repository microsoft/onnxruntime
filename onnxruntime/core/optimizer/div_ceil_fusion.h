// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class DivCeilFusion

Graph transformer that fuses Div -> Ceil into a single DivCeil node.
This fixes a precision issue on WebGPU where f32 division can introduce
small ULP errors that cause Ceil to round up incorrectly
(e.g., 165.0/15.0 = 11.000001 -> Ceil = 12 instead of 11).
*/
class DivCeilFusion : public GraphTransformer {
 public:
  DivCeilFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("DivCeilFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
