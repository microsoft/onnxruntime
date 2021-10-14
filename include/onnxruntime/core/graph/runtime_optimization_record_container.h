// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/graph/runtime_optimization_record.h"

namespace onnxruntime {

class RuntimeOptimizationRecordContainer {
 public:
#if defined(ORT_ENABLE_ADDING_RUNTIME_OPTIMIZATION_RECORDS)
  bool AddRecord(const std::string& sat_key, RuntimeOptimizationRecord&& runtime_optimization_record);
#endif

  std::vector<RuntimeOptimizationRecord> RemoveRecordsForKey(const std::string& sat_key);

 private:
  std::unordered_map<std::string, std::vector<RuntimeOptimizationRecord>> sat_to_optimizations_;
};

}  // namespace onnxruntime
