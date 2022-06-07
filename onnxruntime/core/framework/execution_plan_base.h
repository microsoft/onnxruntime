// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <set>
#include "core/framework/allocator.h"

namespace onnxruntime {

class ExecutionPlanBase {
 public:
  /**
   * Get memory location for a given MLValue
   * @param ort_value_index The index of the mlvalue
   * @return memory location
   */
  virtual const struct OrtMemoryInfo& GetLocation(size_t ort_value_index) const = 0;
  virtual void SetLocation(size_t ort_value_index, const struct OrtMemoryInfo&) = 0;
  // return all memory locations for all the MLValues
  virtual std::set<struct OrtMemoryInfo> GetAllLocations() const = 0;
  virtual ~ExecutionPlanBase() = default;
};

}  // namespace onnxruntime
