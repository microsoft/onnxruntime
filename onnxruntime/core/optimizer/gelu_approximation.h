// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class GeluApproximation

Rewrite graph to replace Gelu or AddGeluFusion by FastGelu node. FastGelu uses approximation for Gelu,
and it is faster.
*/
class GeluApproximation : public GraphTransformer {
 public:
  GeluApproximation(const std::unordered_set<std::string>& compatible_execution_providers={}) noexcept
      : GraphTransformer("GeluApproximation", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
