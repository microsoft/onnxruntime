// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class ReshapeFusion
Rewrite graph fusing reshape subgraph to a single Reshape node.

Before fusion:
   [Sub-graph  Root  Node ]
    |        /            \
    |    Shape            Shape
    |       |                |                 (one or two constant Initializers)
    |    Gather(indice=0)  Gather(indice=1)    int64 a[]  int64 b[] (optional)
    |        \            /                     /         /
    |         \         /                      /         /
    |          \       /   /--------------------        /
    |           Concat    /----------------------------
     \        /
      Reshape

After fusion:
    [Sub-graph  Root  Node ]    (Constant Initializers, b is optional)
                 |                 [0, 0, a, b]
                  \               /
                     Reshape
*/
class ReshapeFusion : public GraphTransformer {
 public:
  ReshapeFusion(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ReshapeFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
