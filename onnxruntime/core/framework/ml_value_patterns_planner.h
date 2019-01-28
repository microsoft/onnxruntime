// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <memory>

#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/framework/mem_pattern_planner.h"
#include "core/framework/allocation_planner.h"

namespace onnxruntime {
struct SequentialExecutionPlan;

class MLValuePatternPlanner {
 public:
  explicit MLValuePatternPlanner(const SequentialExecutionPlan& execution_plan);

  common::Status TraceAllocation(int ml_value_idx, size_t size) {
    auto location = execution_planner_.allocation_plan[ml_value_idx].location;
    auto it = planner_map_.find(location);
    if (it == planner_map_.end()) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
    }

    std::lock_guard<OrtMutex> lock(lock_);
    it->second->TraceAllocation(ml_value_idx, size);
    return common::Status::OK();
  }

  common::Status TraceFree(int ml_value_index) {
    auto location = execution_planner_.allocation_plan[ml_value_index].location;
    auto it = planner_map_.find(location);
    if (it == planner_map_.end()) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
    }

    std::lock_guard<OrtMutex> lock(lock_);
    it->second->TraceFree(ml_value_index);
    return common::Status::OK();
  }

  common::Status GeneratePatterns(MemoryPatternGroup* out) {
    if (!out)
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);

    std::lock_guard<OrtMutex> lock(lock_);
    for (auto& it : planner_map_) {
      out->locations.push_back(it.first);
      out->patterns.push_back(it.second->GenerateMemPattern());
    }

    return common::Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MLValuePatternPlanner);

  mutable OrtMutex lock_;
  std::map<OrtAllocatorInfo, MemPatternPlanner*> planner_map_;
  std::vector<std::unique_ptr<MemPatternPlanner> > pattern_planners_;
  const SequentialExecutionPlan& execution_planner_;
};
}  // namespace onnxruntime
