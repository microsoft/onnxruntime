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
  static bool Fuse_Subgraph2(Node& reshape, Graph& graph, const logging::Logger& logger);
  static void Remove_Unused_nodes(Graph& graph, const std::vector<Node*> nodes_to_remove);
  static bool Replace_Node_With_Initializer(Graph& graph,
                                            const Node& node_to_replace,
                                            const std::vector<int64_t> new_shape,
                                            const logging::Logger& logger);
};

}  // namespace onnxruntime
