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
  static bool Fuse_Subgraph(Node& reshape, Graph& graph, const logging::Logger& logger);
  static bool Match_One_Element_Output_Subgraph_1(Graph& graph, const NodeArg& root_input, const Node& concat,
                                                  int index, std::vector<int64_t> shape_value, bool checkOneElementOnly, const logging::Logger& looger);
  static bool Match_One_Element_Output_Subgraph_2(Graph& graph, const NodeArg& root_input, const Node& concat,
                                                  int index, const logging::Logger& looger);
  static bool Is_One_Element_Input(const Node& cur_node, int index);
  static bool Is_One_Element_Output_Subgraph(Graph& graph, const NodeArg& root_input, const Node& concat,
                                             int index, std::vector<int64_t> shape_value, const logging::Logger& logger);
};

}  // namespace onnxruntime
