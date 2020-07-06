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
  std::vector<NodeArg*> IdentifyTransformerLayerEdges(Graph& graph) const;

  std::vector<Node*> NodesBetweenEdges(const NodeArg* begin, const NodeArg* end) const;

  void InsertRecomputeNode(const Node* node);
};

}  // namespace onnxruntime
