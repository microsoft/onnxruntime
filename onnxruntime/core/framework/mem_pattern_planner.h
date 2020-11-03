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

  // Returns true if there is an intersection between two time schedules.
  // ASSUMES EACH TIME SCHEDULE IS SORTED. THIS IS VALIDATED AT THE END OF MEMORY PLANNING.
  bool OverlappingTimeSchedules(const std::vector<size_t>& program_counter_start_1, const std::vector<size_t>& program_counter_end_1,
                                const std::vector<size_t>& program_counter_start_2, const std::vector<size_t>& program_counter_end_2) {
    ORT_ENFORCE(program_counter_start_1.size() > 0);
    ORT_ENFORCE(program_counter_start_2.size() > 0);
    ORT_ENFORCE(program_counter_start_1.size() == program_counter_end_1.size());
    ORT_ENFORCE(program_counter_start_2.size() == program_counter_end_2.size());

    size_t index_1 = 0;
    size_t index_2 = 0;
    while ((index_1 < program_counter_start_1.size()) && (index_2 < program_counter_start_2.size())) {
      if (program_counter_start_1[index_1] <= program_counter_start_2[index_2]) {
        if (program_counter_end_1[index_1] >= program_counter_start_2[index_2]) {
          return true;
        }
        index_1 += 1;
      } else {
        if (program_counter_end_2[index_2] >= program_counter_start_1[index_1]) {
          return true;
        }
        index_2 += 1;
      }
    }

    return false;
  }

  void TraceAllocation(int ml_value_idx, const std::vector<size_t>& program_counter_start, const std::vector<size_t>& program_counter_end, size_t size) {
    std::lock_guard<OrtMutex> lock(lock_);

    if (size == 0) {
      allocs_.emplace_back(ml_value_idx, MemoryBlock(0, 0));
      return;
    }

    size_t current = 0;
    size_t waste_bytes = std::numeric_limits<size_t>::max();
    size_t best_offset = 0;
    bool best_offset_found = false;
    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      // Memory block can be re-used as long as there is no overlap between their time schedules.
      if (allocs_[*it].reuse_ && !OverlappingTimeSchedules(program_counter_start, program_counter_end,
                                                           allocs_[*it].program_counter_start_, allocs_[*it].program_counter_end_)) {
        continue;
      }

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
    allocs_.emplace_back(ml_value_idx, program_counter_start, program_counter_end, MemoryBlock(best_offset, size));
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

  void TraceAllocation(int ml_value_idx, size_t size) {
    std::lock_guard<OrtMutex> lock(lock_);

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
    std::lock_guard<OrtMutex> lock(lock_);

    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].index_ == ml_value_index) {
        blocks_.erase(it);
        break;
      }
    }
  }

  MemoryPattern GenerateMemPattern() {
    std::lock_guard<OrtMutex> lock(lock_);

    // Time schedules of overlapping memory blocks SHOULD NOT intersect.
    for (size_t index_1 = 0; index_1 < allocs_.size(); index_1 += 1) {
      if (!allocs_[index_1].reuse_)
        continue;

      for (size_t index_2 = index_1 + 1; index_2 < allocs_.size(); index_2 += 1) {
        if (!allocs_[index_2].reuse_)
          continue;

        size_t alloc_1_start = allocs_[index_1].block_.offset_;
        size_t alloc_1_end = alloc_1_start + allocs_[index_1].block_.size_ - 1;

        ORT_ENFORCE(alloc_1_start <= alloc_1_end);

        size_t alloc_2_start = allocs_[index_2].block_.offset_;
        size_t alloc_2_end = alloc_2_start + allocs_[index_2].block_.size_ - 1;

        ORT_ENFORCE(alloc_2_start <= alloc_2_end);

        if (((alloc_1_start >= alloc_2_start) && (alloc_1_start <= alloc_2_end)) ||
            ((alloc_2_start >= alloc_1_start) && (alloc_2_start <= alloc_1_end))) {
          ORT_ENFORCE(!OverlappingTimeSchedules(allocs_[index_1].program_counter_start_, allocs_[index_1].program_counter_end_,
                                                allocs_[index_2].program_counter_start_, allocs_[index_2].program_counter_end_));
        }
      }
    }

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
    const std::vector<size_t> program_counter_start_;
    const std::vector<size_t> program_counter_end_;
    bool reuse_{false};
    OrtValueAllocationBlock() = default;
    OrtValueAllocationBlock(int index, const MemoryBlock& block) : index_(index), block_(block), reuse_{false} {}
    OrtValueAllocationBlock(int index, std::vector<size_t> program_counter_start, std::vector<size_t> program_counter_end, const MemoryBlock& block) : index_(index), block_(block), program_counter_start_(program_counter_start), program_counter_end_(program_counter_end), reuse_{true} {}
  };

  std::vector<OrtValueAllocationBlock> allocs_;
  // blocks_ the list of currently allocated memory blocks, sorted in order of their offset
  std::list<int> blocks_;
  SafeInt<size_t> buffer_size_{0};
  mutable OrtMutex lock_;
};

}  // namespace onnxruntime
