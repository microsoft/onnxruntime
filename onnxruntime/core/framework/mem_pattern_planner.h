// Part of the algo is derived from tensorflow.

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <algorithm>
#include <limits>
#include <list>
#include <mutex>
#include <vector>
#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/mem_pattern.h"

namespace onnxruntime {
// MemPatternPlanner is used to trace allocation/free steps
// in a single iteration, record the pattern and cached for
// future request if they have the same input shape.
// Thread-safe.
class MemPatternPlanner {
 public:
  void TraceAllocation(int ml_value_idx, size_t size) {
    std::lock_guard<std::mutex> lock(lock_);

    if (size == 0) {
      allocs_.emplace_back(ml_value_idx, MemoryBlock(0, 0));
      return;
    }

    size_t current = 0;
    size_t waste_bytes = std::numeric_limits<size_t>::max();
    size_t best_offset = 0;
    bool best_offset_found = false;

    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].block_.offset_ >= current) {
        auto gap = allocs_[*it].block_.offset_ - current;
        if (gap >= size && (gap - size) < waste_bytes) {
          waste_bytes = gap - size;
          best_offset = current;
          best_offset_found = true;
        }
      }
      current = std::max(current, allocs_[*it].block_.offset_ + allocs_[*it].block_.size_);
    }

    ORT_ENFORCE(current <= buffer_size_);

    if (current < buffer_size_) {
      auto gap = buffer_size_ - current;
      if ((gap >= size) && ((gap - size) < waste_bytes)) {
        best_offset = current;
        best_offset_found = true;
      }
    }

    if (!best_offset_found) {
      best_offset = current;
    }

    // we only need to bounds check the addition of size to best_offset as that is the only time we extend
    // the maximum size of the buffer.
    buffer_size_ = std::max(buffer_size_, SafeInt<size_t>(best_offset) + size);
    allocs_.emplace_back(ml_value_idx, MemoryBlock(best_offset, size));
    std::list<int>::iterator best_fit_it = blocks_.end();
    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].block_.offset_ < best_offset)
        continue;

      if ((allocs_[*it].block_.offset_ > best_offset) || (allocs_[*it].block_.size_ >= size)) {
        best_fit_it = it;
        break;
      }
    }

    blocks_.insert(best_fit_it, (static_cast<int>(allocs_.size()) - 1));
  }

  void TraceFree(int ml_value_index) {
    std::lock_guard<std::mutex> lock(lock_);

    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].index_ == ml_value_index) {
        blocks_.erase(it);
        break;
      }
    }
  }

  MemoryPattern GenerateMemPattern() const {
    std::lock_guard<std::mutex> lock(lock_);

    MemoryPattern pattern;
    pattern.peak_size_ = buffer_size_;
    pattern.patterns_.reserve(allocs_.size());
    for (auto& alloc : allocs_) {
      pattern.patterns_.insert_or_assign(alloc.index_, alloc.block_);
    }

    return pattern;
  }

 private:
  struct OrtValueAllocationBlock {
    int index_{-1};
    MemoryBlock block_;
    OrtValueAllocationBlock() = default;
    OrtValueAllocationBlock(int index, const MemoryBlock& block) : index_(index), block_(block) {}
  };

  std::vector<OrtValueAllocationBlock> allocs_;
  // blocks_ the list of currently allocated memory blocks, sorted in order of their offset
  std::list<int> blocks_;
  SafeInt<size_t> buffer_size_{0};
  mutable std::mutex lock_;
};

}  // namespace onnxruntime
