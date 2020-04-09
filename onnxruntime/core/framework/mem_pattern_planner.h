//Part of the algo is derived from tensorflow.

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
#include <list>
#include "core/common/safeint.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/allocation_planner.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
// MemPatternPlanner is used to trace allocation/free steps
// in a single iteration, record the pattern and cached for
// future request if they have the same input shape.
// Thread-safe.
class MemPatternPlanner {
 public:
  MemPatternPlanner() = default;

  void TraceAllocation(int ml_value_idx, size_t size) {
    std::lock_guard<OrtMutex> lock(lock_);

    if (size == 0) {
      allocs_.emplace_back(ml_value_idx, MemoryBlock(0, 0));
      return;
    }

    size_t current = 0;
    size_t waste_bytes = std::numeric_limits<size_t>::max();
    size_t best_offset = 0;
    if (!blocks_.empty()) {
      auto last_block = allocs_[*blocks_.rbegin()];
      best_offset = last_block.block_.offset_ + last_block.block_.size_;
    }

    std::list<int>::iterator best_fit_it = blocks_.end();
    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].block_.offset_ >= current) {
        auto gap = allocs_[*it].block_.offset_ - current;
        if (gap >= size && (gap - size) < waste_bytes) {
          best_fit_it = it;
          waste_bytes = gap - size;
          best_offset = current;
        }
      }
      current = allocs_[*it].block_.offset_ + allocs_[*it].block_.size_;
    }

    // we only need to bounds check the addition of size to best_offset as that is the only time we extend
    // the maximum size of the buffer.
    buffer_size_ = std::max(buffer_size_, SafeInt<size_t>(best_offset) + size);
    allocs_.emplace_back(ml_value_idx, MemoryBlock(best_offset, size));
    blocks_.insert(best_fit_it, (static_cast<int>(allocs_.size()) - 1));
  }

  void TraceFree(int ml_value_index) {
    std::lock_guard<OrtMutex> lock(lock_);

    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].index_ == ml_value_index) {
        blocks_.erase(it);
        break;
      }
    }
  }

  MemoryPattern GenerateMemPattern() const {
    std::lock_guard<OrtMutex> lock(lock_);

    MemoryPattern pattern;
    pattern.peak_size_ = buffer_size_;
    for (auto& alloc : allocs_) {
      pattern.patterns_[alloc.index_] = alloc.block_;
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
  mutable OrtMutex lock_;
};

}  // namespace onnxruntime
