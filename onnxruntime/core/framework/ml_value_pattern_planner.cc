// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <set>
#include "core/framework/ml_value_patterns_planner.h"
#include "core/framework/sequential_execution_plan.h"

namespace onnxruntime {
MLValuePatternPlanner::MLValuePatternPlanner(const SequentialExecutionPlan& execution_plan)
    : execution_planner_{execution_plan} {
  std::set<OrtAllocatorInfo> locations;
  for (auto& alloc_plan : execution_planner_.allocation_plan) {
    if (locations.find(alloc_plan.location) == locations.end())
      locations.insert(alloc_plan.location);
  }
  for (auto& location : locations) {
    pattern_planners_.push_back(std::make_unique<MemPatternPlanner>());
    planner_map_[location] = pattern_planners_.back().get();
  }
}
}  // namespace onnxruntime
