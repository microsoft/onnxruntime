// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

class TransformerLayerRecompute : public GraphTransformer {
 public:
  TransformerLayerRecompute(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("TransformerLayerRecompute", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  Status IdentifyTransformerLayerEdges(Graph& graph, std::vector<std::pair<const NodeArg*, const NodeArg*>>& start_end_edges) const;

  std::vector<const Node*> NodesBetweenEdges(Graph& graph, const NodeArg* start, const NodeArg* end) const;

  void InsertRecomputeNodes(Graph& graph, const std::vector<const Node*>& nodes) const;
};

}  // namespace onnxruntime
