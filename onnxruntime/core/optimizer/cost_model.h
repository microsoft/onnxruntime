// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/op_runtime_profiler.h"
#include "core/graph/graph.h"

namespace onnxruntime {

class CostModel {
public:
  explicit CostModel(std::shared_ptr<OpRuntimeProfiler> op_runtime_profiler)
      : op_runtime_profiler_(op_runtime_profiler) {}

  float ComputeCost(const Graph& graph);

  static const int max_cost = 1e9;

private:
  float ComputeThroughput(const Graph& graph);
  float ComputeMemory(const Graph& graph);

  std::shared_ptr<OpRuntimeProfiler> op_runtime_profiler_;
};

}  // namespace onnxruntime
