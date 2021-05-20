// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/common.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
// The interface for arena which manage memory allocations
// Arena will hold a pool of pre-allocate memories and manage their lifecycle.
// Need an underline IResourceAllocator to allocate memories.
// The setting like max_chunk_size is init by IDeviceDescriptor from resource allocator
class IArenaAllocator : public IAllocator {
 public:
  IArenaAllocator(const OrtMemoryInfo& info) : IAllocator(info) {}
  ~IArenaAllocator() override = default;
  // Alloc call needs to be thread safe.
  void* Alloc(size_t size) override = 0;
  // The chunk allocated by Reserve call won't be reused with other request
  // (i.e.) it is not maintained by the arena and
  // it will be return to the devices when it is freed.
  // Reserve call needs to be thread safe.
  virtual void* Reserve(size_t size) = 0;
  // Free call needs to be thread safe.
  void Free(void* p) override = 0;
  // All unused device allocations maintained by the arena
  // (i.e.) physical allocations with no chunks in use will be de-allocated.
  // Shrink call needs to be thread safe.
  virtual Status Shrink() = 0;
  virtual size_t Used() const = 0;
  virtual size_t Max() const = 0;
  // allocate host pinned memory?
};

using ArenaPtr = std::shared_ptr<IArenaAllocator>;

// Runtime statistics collected by an allocator.
struct AllocatorStats {
  int64_t num_allocs;             // Number of allocations.
  int64_t num_reserves;           // Number of reserves. (Number of calls to Reserve() in arena-based allocators)
  int64_t num_arena_extensions;   // Number of arena extensions (Relevant only for arena based allocators)
  int64_t num_arena_shrinkages;   // Number of arena shrinkages (Relevant only for arena based allocators)
  int64_t bytes_in_use;           // Number of bytes in use.
  int64_t total_allocated_bytes;  // The total number of allocated bytes by the allocator.
  int64_t max_bytes_in_use;       // The maximum bytes in use.
  int64_t max_alloc_size;         // The max single allocation seen.
                                  // The upper limit what the allocator can allocate, if such a limit
                                  // is known. Certain allocator may return 0 to indicate the limit is
                                  // unknown.
  int64_t bytes_limit;

  AllocatorStats() { Clear(); }

  void Clear() {
    this->num_allocs = 0;
    this->num_reserves = 0;
    this->num_arena_extensions = 0;
    this->num_arena_shrinkages = 0;
    this->bytes_in_use = 0;
    this->max_bytes_in_use = 0;
    this->max_alloc_size = 0;
    this->bytes_limit = 0;
    this->total_allocated_bytes = 0;
  }

  std::string DebugString() const {
    std::ostringstream ss;
    ss << "Limit:                    " << this->bytes_limit << "\n"
       << "InUse:                    " << this->bytes_in_use << "\n"
       << "TotalAllocated:           " << this->total_allocated_bytes << "\n"
       << "MaxInUse:                 " << this->max_bytes_in_use << "\n"
       << "NumAllocs:                " << this->num_allocs << "\n"
       << "NumReserves:              " << this->num_reserves << "\n"
       << "NumArenaExtensions:       " << this->num_arena_extensions << "\n"
       << "NumArenaShrinkages:       " << this->num_arena_shrinkages << "\n"
       << "MaxAllocSize:             " << this->max_alloc_size << "\n";
    return ss.str();
  }
};
}  // namespace onnxruntime