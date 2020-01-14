// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

<<<<<<< HEAD
class GeluFusion : public GraphTransformer {
 public:
  GeluFusion(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept 
      : GraphTransformer("GeluFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;
=======
/**
@Class GeluFusion

Rewrite graph fusing Gelu activation subgraph to a single Gelu node.

The formula corresponding to Gelu activation subgraph:
x * 0.5 * (1.0 + erf(x / sqrt(2.0))), where x is the input.

*/
class GeluFusion : public GraphTransformer {
 public:
  GeluFusion(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("GeluFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677
};

}  // namespace onnxruntime
