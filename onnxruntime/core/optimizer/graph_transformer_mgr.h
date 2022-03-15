// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

// Manages a list of graph transformers. It is initialized with a list of graph
// transformers. Each inference session can further register additional ones.
class GraphTransformerManager {
 public:
  explicit GraphTransformerManager(unsigned steps) : steps_(steps) {
  }

  // Update (set) the maximum number of graph transformation steps
  common::Status SetSteps(unsigned steps);

  // Get the maximum number of graph transformation steps
  common::Status GetSteps(unsigned& steps) const;

  // Register a transformer with a level.
  common::Status Register(std::unique_ptr<GraphTransformer> transformer, TransformerLevel level);

  // Apply all transformers registered for the given level on the given graph
  common::Status ApplyTransformers(Graph& graph, TransformerLevel level, const logging::Logger& logger) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformerManager);

  // maximum number of graph transformation steps
  unsigned steps_;

  InlinedHashMap<TransformerLevel, InlinedVector<std::unique_ptr<GraphTransformer>>> level_to_transformer_map_;
  InlinedHashMap<std::string, GraphTransformer*> transformers_info_;
};
}  // namespace onnxruntime
