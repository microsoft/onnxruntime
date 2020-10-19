// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/cost_model.h"
#include "core/optimizer/op_runtime_profiler.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

float CostModel::ComputeCost(const Graph& graph) {
  // TODO: Compute max memory using topology
  float max_memory = 1e10;
  float memory_usage = ComputeMemory(graph);
  if (memory_usage > max_memory) {
    return max_cost;
  }

  float throughput = ComputeThroughput(graph);
  return -throughput;
}

float CostModel::ComputeThroughput(const Graph& graph) {
  // TODO: Implement
  (void) graph;
  return 0;
}

float CostModel::ComputeMemory(const Graph& graph) {
  // TODO: Implement
  (void) graph;
  return 0;
}

} // namespace onnxruntime
