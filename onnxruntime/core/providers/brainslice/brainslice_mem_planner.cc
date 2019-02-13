#include "core/providers/brainslice/brainslice_mem_planner.h"

namespace onnxruntime {

BrainSliceMemoryPlanner::BrainSliceMemoryPlanner(ISA_Mem mem_type, size_t capacity, int start_address) : mem_type_(mem_type),
                                                                                                         capacity_(capacity),
                                                                                                         start_address_(start_address),
                                                                                                         current_block_idx_(0) {
}

int BrainSliceMemoryPlanner::Alloc(size_t size) {
  // Try allocate first, if exceed capacity, free it
  TraceAllocation(current_block_idx_, size);
  if (buffer_size > capacity_) {
    TraceFree(current_block_idx_);
    return -1;
  }

  int offset = -1;
  for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
    if (allocs_[*it].index_ == current_block_idx_) {
      offset = static_cast<int>(allocs_[*it].block_.offset_);
      break;
    }
  }
  //Here the allocation succeed, we should always get a valid offset.
  ORT_ENFORCE(offset >= 0);
  address_to_block_[offset] = current_block_idx_++;
  return offset + start_address_;
}

void BrainSliceMemoryPlanner::Free(int address) {
  int offset = address - start_address_;
  auto it = address_to_block_.find(offset);
  if (it != address_to_block_.end()) {
    TraceFree(it->second);
    address_to_block_.erase(it);
  }
}
}  // namespace onnxruntime
