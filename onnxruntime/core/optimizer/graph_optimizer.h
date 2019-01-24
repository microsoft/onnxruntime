// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

class GraphOptimizer {
 public:
  GraphOptimizer(Graph& graph, logging::Logger& logger) : graph_(graph),
                                                          logger_(logger),
                                                          session_state_(execution_providers_) {
  }

  Status Init();

private:
  Graph& graph_;
  logging::Logger& logger_;

  ExecutionProviders execution_providers_;
  SessionState session_state_;
};

}  // namespace onnxruntime