// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <unordered_map>

#include "core/common/common.h"

// This class is not thread-safe
// TODO: this is a static hash lookup, it's easy to do it better
namespace onnxruntime {
class MLValueNameIdxMap {
 public:
  using const_iterator = typename std::unordered_map<std::string, int>::const_iterator;

  MLValueNameIdxMap() = default;

  // Add OrtValue name to map and return index associated with it.
  // If entry already existed the existing index value is returned.
  int Add(const std::string& name) {
    auto it = map_.find(name);
    if (it == map_.end()) {
      int idx;
      idx = ort_value_max_idx_++;
      map_.insert(it, {name, idx});
      return idx;
    }
    return it->second;
  }

  common::Status GetIdx(const std::string& name, int& idx) const {
    idx = -1;

    auto it = map_.find(name);
    if (it == map_.end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Could not find OrtValue with name '", name, "'");
    }

    idx = it->second;
    return common::Status::OK();
  }

  size_t Size() const { return map_.size(); };
  int MaxIdx() const { return ort_value_max_idx_; }

  const_iterator begin() const noexcept { return map_.cbegin(); }
  const_iterator end() const noexcept { return map_.cend(); }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MLValueNameIdxMap);

  int ort_value_max_idx_ = 0;
  std::unordered_map<std::string, int> map_;
};
using OrtValueNameIdxMap = MLValueNameIdxMap;
}  // namespace onnxruntime
