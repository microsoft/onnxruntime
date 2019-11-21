// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class ReshapeFusion
Rewrite graph fusing reshape subgraph to a single Reshape node.
*/
class ReshapeFusion : public GraphTransformer {
 public:
  ReshapeFusion(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ReshapeFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

private:
  static bool Fuse_Subgraph1(Node& reshape, Graph& graph, const logging::Logger& logger);
};

}  // namespace onnxruntime
