// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/allocation_planner.h"

namespace onnxruntime {
struct MemoryBlock {
  size_t offset_{0};
  size_t size_{0};

  MemoryBlock() = default;
  MemoryBlock(size_t offset, size_t size) : offset_(offset), size_(size) {}
};

class MemoryPattern {
  friend class MemPatternPlanner;

 public:
  MemoryPattern() = default;

  MemoryPattern(MemoryPattern&& rhs) noexcept
      : patterns_{std::move(rhs.patterns_)},
        peak_size_{std::move(rhs.peak_size_)} {}

  MemoryPattern& operator=(MemoryPattern&& rhs) noexcept {
    patterns_ = std::move(rhs.patterns_);
    peak_size_ = std::move(rhs.peak_size_);
    return *this;
  }

  size_t PeakSize() const {
    return peak_size_;
  }

  const MemoryBlock* GetBlock(int ml_value_idx) const {
    auto it = patterns_.find(ml_value_idx);
    if (it == patterns_.end())
      return nullptr;

    return &it->second;
  }

  // REVIEW (codemzs): Put some mechanism in place to ensure the sanity of the pattern when it is 
  // amended, i.e integrity of peak size.
  void InsertBlock(int ml_value_idx, MemoryBlock block) {

    ORT_ENFORCE(patterns_.find(ml_value_idx) == patterns_.end());

    patterns_[ml_value_idx] = block;
  }

  void EraseBlock(int ml_value_idx) {

    ORT_ENFORCE(patterns_.find(ml_value_idx) != patterns_.end());

    patterns_.erase(ml_value_idx);
  }

 private:
  // allow move
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(MemoryPattern);

  std::unordered_map<int, MemoryBlock> patterns_;
  size_t peak_size_{0};
};

struct MemoryPatternGroup {
  std::vector<OrtMemoryInfo> locations;
  std::vector<MemoryPattern> patterns;

  const MemoryPattern* GetPatterns(const OrtMemoryInfo& location) const {
    for (size_t i = 0; i < locations.size(); i++)
      if (locations[i] == location) {
        return &patterns[i];
      }
    return nullptr;
  }
};
}  // namespace onnxruntime
