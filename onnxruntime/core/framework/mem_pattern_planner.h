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

  void TraceAllocation(int ml_value_idx, std::vector<size_t> program_counter_start, std::vector<size_t> program_counter_end, size_t size) {
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

    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
            // Memory block can be re-used as long as there is no overlap between their time schedules.
      bool overlap = false;
      if (allocs_[*it].reuse_) {
        //bool keep_looping = true;
        //while(keep_looping);
        ORT_ENFORCE(program_counter_start.size() == program_counter_end.size());
        ORT_ENFORCE(program_counter_start.size() == program_counter_end.size());

        // Ensure memory time schedule is sorted.
        size_t start = 0;
        for (size_t index = 0; index < program_counter_start.size(); index += 1) {
          ORT_ENFORCE((program_counter_start[index] > start) || (start == 0));
          ORT_ENFORCE(program_counter_start[index] <= program_counter_end[index]);
          start = program_counter_start[index];
        }

        size_t index_allocated = 0;
        size_t index_to_be_allocated = 0;
        while ((index_allocated < allocs_[*it].program_counter_start_.size()) && (index_to_be_allocated < program_counter_start.size())) {
          if (allocs_[*it].program_counter_start_[index_allocated] <= program_counter_start[index_to_be_allocated]) {
            if (allocs_[*it].program_counter_end_[index_allocated] >= program_counter_start[index_to_be_allocated]) {
              overlap = true;
              break;
            }
            index_allocated += 1;
          } else {
            if (program_counter_end[index_to_be_allocated] >= allocs_[*it].program_counter_start_[index_allocated]) {
              overlap = true;
              break;
            }
            index_to_be_allocated += 1;
          }
        }
      }

      //if (allocs_[*it].reuse_ && ((allocs_[*it].program_counter_start_ > program_counter_end) || (allocs_[*it].program_counter_end_ < program_counter_start)))
      //continue;

      if (allocs_[*it].reuse_ && !overlap)
        continue;

      if (allocs_[*it].block_.offset_ >= current) {
        auto gap = allocs_[*it].block_.offset_ - current;
        if (gap >= size && (gap - size) < waste_bytes) {
          waste_bytes = gap - size;
          best_offset = current;
        }
      }
      current = allocs_[*it].block_.offset_ + allocs_[*it].block_.size_;
    }

    // we only need to bounds check the addition of size to best_offset as that is the only time we extend
    // the maximum size of the buffer.
    buffer_size_ = std::max(buffer_size_, SafeInt<size_t>(best_offset) + size);
    allocs_.emplace_back(ml_value_idx, program_counter_start, program_counter_end, MemoryBlock(best_offset, size));
    std::list<int>::iterator best_fit_it = blocks_.end();
    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].block_.offset_ >= best_offset) {
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
    std::vector<size_t> program_counter_start_;
    std::vector<size_t> program_counter_end_;
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
