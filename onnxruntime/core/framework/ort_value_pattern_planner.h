// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <memory>

#include "core/common/common.h"
#include "core/framework/mem_pattern_planner.h"
#include "core/framework/execution_plan_base.h"
#include "core/framework/allocation_planner.h"

namespace onnxruntime {
class ExecutionPlanBase;

// Thread-safe
// As it doesn't always work, the usage of it must be guarded by
// SessionOptions.enable_mem_pattern
class OrtValuePatternPlanner {
 public:
  explicit OrtValuePatternPlanner(const ExecutionPlanBase& execution_plan);
  common::Status TraceAllocation(int ort_value_idx, const std::vector<size_t>& program_counter_start, const std::vector<size_t>& program_counter_end, size_t size);
  common::Status TraceAllocation(int ort_value_idx, size_t size);
  common::Status TraceFree(int ort_value_index);
  common::Status GeneratePatterns(MemoryPatternGroup* out);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrtValuePatternPlanner);

 private:
  // This map itself is const after the construction
  std::map<OrtMemoryInfo, std::unique_ptr<MemPatternPlanner>> planner_map_;
  const ExecutionPlanBase& execution_planner_;
};
}  // namespace onnxruntime
