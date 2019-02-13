// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/framework/mem_pattern_planner.h"
#include "core/providers/brainslice/fpga_handle.h"

namespace onnxruntime {

class BrainSliceMemoryPlanner : public MemPatternPlanner {
 public:
  BrainSliceMemoryPlanner(ISA_Mem mem_type, size_t capacity, int start_address);

  int Alloc(size_t size);

  void Free(int address);

 private:
  ISA_Mem mem_type_;
  size_t capacity_;
  int start_address_;
  int current_block_idx_;
  std::unordered_map<int, int> address_to_block_;
};
}  // namespace onnxruntime
