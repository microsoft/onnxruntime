// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../plugin_ep_utils.h"

#include <functional>
#include <memory>

// `OrtAllocator` is a C API struct. `BaseAllocator` is a minimal C++ struct which inherits from `OrtAllocator`.
// Notably, `BaseAllocator` has a virtual destructor to enable a derived class to be deleted through a `BaseAllocator`
// pointer. Allocators which need to be deleted through a base class pointer should inherit from `BaseAllocator`.
struct BaseAllocator : OrtAllocator {
  virtual ~BaseAllocator() = default;
};

using AllocatorUniquePtr = std::unique_ptr<BaseAllocator>;

struct CustomAllocator : BaseAllocator {
  CustomAllocator(const OrtMemoryInfo* mem_info) : memory_info{mem_info} {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;  // no special reserve logic and most likely unnecessary unless you have your own arena
    GetStats = nullptr;
    AllocOnStream = nullptr;
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* /*this_*/, size_t size) {
    return malloc(size);
  }

  /// Free a block of memory previously allocated with OrtAllocator::Alloc
  static void ORT_API_CALL FreeImpl(struct OrtAllocator* /*this_*/, void* p) {
    return free(p);
  }

  /// Return a pointer to an ::OrtMemoryInfo that describes this allocator
  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const CustomAllocator& impl = *static_cast<const CustomAllocator*>(this_);
    return impl.memory_info;
  }

 private:
  const OrtMemoryInfo* memory_info;
};

using AllocationUniquePtr = std::unique_ptr<void, std::function<void(void*)>>;

inline AllocationUniquePtr AllocateBytes(OrtAllocator* allocator, size_t num_bytes) {
  void* p = allocator->Alloc(allocator, num_bytes);
  return AllocationUniquePtr(p, [allocator](void* d) { allocator->Free(allocator, d); });
}
