// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/runtime_optimization_record_container.h"

#include <algorithm>

namespace onnxruntime {

#if defined(ORT_ENABLE_ADDING_RUNTIME_OPTIMIZATION_RECORDS)
bool RuntimeOptimizationRecordContainer::AddRecord(const std::string& sat_key,
                                                   RuntimeOptimizationRecord&& runtime_optimization_record) {
  auto& optimizations = sat_to_optimizations_[sat_key];
  if (std::find(optimizations.begin(), optimizations.end(), runtime_optimization_record) != optimizations.end()) {
    return false;
  }

  optimizations.emplace_back(std::move(runtime_optimization_record));
  return true;
}
#endif

std::vector<RuntimeOptimizationRecord> RuntimeOptimizationRecordContainer::RemoveRecordsForKey(
    const std::string& sat_key) {
  std::vector<RuntimeOptimizationRecord> result{};
  if (auto it = sat_to_optimizations_.find(sat_key); it != sat_to_optimizations_.end()) {
    result = std::move(it->second);
    sat_to_optimizations_.erase(it);
  }
  return result;
}

}  // namespace onnxruntime
