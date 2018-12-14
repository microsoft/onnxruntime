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
  const OrtAllocatorInfo& Info() const override = 0;
  // allocate host pinned memory?
};

using ArenaPtr = std::shared_ptr<IArenaAllocator>;

// Dummy Arena which just call underline device allocator directly.
class DummyArena : public IArenaAllocator {
 public:
  explicit DummyArena(std::unique_ptr<IDeviceAllocator> resource_allocator)
      : allocator_(std::move(resource_allocator)),
        info_(allocator_->Info().name, OrtAllocatorType::OrtArenaAllocator, allocator_->Info().id) {
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

  const OrtAllocatorInfo& Info() const override {
    return info_;
  }

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DummyArena);

  std::unique_ptr<IDeviceAllocator> allocator_;
  OrtAllocatorInfo info_;
};
}  // namespace onnxruntime
