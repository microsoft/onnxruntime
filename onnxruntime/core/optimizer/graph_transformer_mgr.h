// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/constant_folding.h"

namespace onnxruntime {
// Manages a list of graph transformers. It is initialized with a list of graph
// transformers. Each inference session can further register additional ones.
class GraphTransformerManager {
 public:
  explicit GraphTransformerManager(unsigned steps) noexcept : steps_(steps) {
    // Register default transformers.
  }

  // Register a graph transformer.
  common::Status Register(std::unique_ptr<GraphTransformer> transformer) {
    transformers_.push_back(std::move(transformer));
    return common::Status::OK();
  }

  // Apply the list of graph transformers registered on the specified graph
  // up to the given number of steps.
  common::Status ApplyAll(Graph& graph) const;

 private:
  GraphTransformerManager() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformerManager);

  std::vector<std::unique_ptr<GraphTransformer>> transformers_;
  const unsigned steps_;
};
}  // namespace onnxruntime
