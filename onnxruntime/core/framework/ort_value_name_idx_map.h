// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <unordered_map>

#include "core/common/common.h"

// This class is not thread-safe
// TODO: this is a static hash lookup, it's easy to do it better
namespace onnxruntime {
class OrtValueNameIdxMap {
 public:
  using const_iterator = typename std::unordered_map<std::string, int>::const_iterator;

  OrtValueNameIdxMap() = default;

  // Add OrtValue name to map and return index associated with it.
  // If entry already existed the existing index value is returned.
  int Add(const std::string& name) {
    auto it = map_.find(name);
    if (it == map_.end()) {
      int idx = next_idx_++;
      map_.insert(it, {name, idx});
      idx_name_map_[idx] = name;
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

  common::Status GetName(int idx, std::string& name) const {
    auto it = idx_name_map_.find(idx);
    if (it == idx_name_map_.end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Could not find OrtValue with idx '", idx, "'");
    }

    name = it->second;
    return common::Status::OK();
  }

  size_t Size() const { return map_.size(); };
  int MaxIdx() const { return next_idx_ - 1; }

  const_iterator begin() const noexcept { return map_.cbegin(); }
  const_iterator end() const noexcept { return map_.cend(); }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrtValueNameIdxMap);

  int next_idx_ = 0;
  std::unordered_map<std::string, int> map_;
  std::unordered_map<int, std::string> idx_name_map_;
};
}  // namespace onnxruntime
