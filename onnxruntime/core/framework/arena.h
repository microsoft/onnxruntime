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
  ~IArenaAllocator() override = default;
  // Alloc call need to be thread safe.
  void* Alloc(size_t size) override = 0;
  // The chunck allocated by Reserve call won't be reused with other request.
  // It will be return to the devices when it is freed.
  // Reserve call need to be thread safe.
  virtual void* Reserve(size_t size) = 0;
  // Free call need to be thread safe.
  void Free(void* p) override = 0;
  virtual size_t Used() const = 0;
  virtual size_t Max() const = 0;
  const OrtMemoryInfo& Info() const override = 0;
  // allocate host pinned memory?
};

using ArenaPtr = std::shared_ptr<IArenaAllocator>;

// Dummy Arena which just call underline device allocator directly.
class DummyArena : public IArenaAllocator {
 public:
  explicit DummyArena(std::unique_ptr<IDeviceAllocator> resource_allocator)
      : allocator_(std::move(resource_allocator)),
        info_(allocator_->Info().name, OrtAllocatorType::OrtArenaAllocator, allocator_->Info().device, allocator_->Info().id) {
  }

  ~DummyArena() override = default;

  void* Alloc(size_t size) override {
    if (size == 0)
      return nullptr;
    return allocator_->Alloc(size);
  }

  void Free(void* p) override {
    allocator_->Free(p);
  }

  void* Reserve(size_t size) override {
    return Alloc(size);
  }

  size_t Used() const override {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  size_t Max() const override {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  const OrtMemoryInfo& Info() const override {
    return info_;
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DummyArena);

  std::unique_ptr<IDeviceAllocator> allocator_;
  OrtMemoryInfo info_;
};

// Runtime statistics collected by an allocator.
struct AllocatorStats {
  int64_t num_allocs;             // Number of allocations.
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
    this->bytes_in_use = 0;
    this->max_bytes_in_use = 0;
    this->max_alloc_size = 0;
    this->bytes_limit = 0;
    this->total_allocated_bytes = 0;
  }

  std::string DebugString() const {
    std::ostringstream ss;
    ss << "Limit:           " << this->bytes_limit << "\n"
       << "InUse:          " << this->bytes_in_use << "\n"
       << "TotalAllocated: " << this->total_allocated_bytes << "\n"
       << "MaxInUse:       " << this->max_bytes_in_use << "\n"
       << "NumAllocs:      " << this->num_allocs << "\n"
       << "MaxAllocSize:   " << this->max_alloc_size << "\n";
    return ss.str();
  }
};
}  // namespace onnxruntime
