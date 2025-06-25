// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "example_plugin_ep_utils.h"

struct CustomAllocator : OrtAllocator {
  CustomAllocator(const OrtMemoryInfo* mem_info) : memory_info{mem_info} {
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;  // no special reserve logic and most likely unnecessary unless you have your own arena
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* /*this_*/, size_t size) {
    // CustomAllocator& impl = *static_cast<CustomAllocator*>(this_);
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
