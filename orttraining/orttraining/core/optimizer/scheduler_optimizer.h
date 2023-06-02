// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <charconv>
#include "core/common/inlined_containers.h"
#include "core/common/string_utils.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MemoryOptimizer

Find recomputable subgraphs and enable according to user configs.
*/

class SchedulerOptimizer : public GraphTransformer {
 public:
  SchedulerOptimizer() : GraphTransformer("SchedulerOptimizer", {}) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }
};

}  // namespace onnxruntime
