// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "example_plugin_ep_utils.h"

// from onnxruntime/core/framework/allocator_stats.h
struct AllocatorStats {
  int64_t num_allocs;             // Number of allocations.
  int64_t num_reserves;           // Number of reserves. (Number of calls to Reserve() in arena-based allocators)
  int64_t bytes_in_use;           // Number of bytes in use.
  int64_t total_allocated_bytes;  // The total number of allocated bytes by the allocator.
  int64_t max_bytes_in_use;       // The maximum bytes in use.
  int64_t max_alloc_size;         // The max single allocation seen.
  int64_t bytes_limit;            // The upper limit what the allocator can allocate, if such a limit
                                  // is known. Certain allocator may return 0 to indicate the limit is
                                  // unknown.
};

struct CustomAllocator : OrtAllocator {
  CustomAllocator(const OrtMemoryInfo* mem_info, const ApiPtrs& api_ptrs_in)
      : memory_info{mem_info}, api_ptrs{api_ptrs_in} {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;      // no special reserve logic and most likely unnecessary unless you have your own arena
    GetStats = GetStatsImpl;  // this can be set to nullptr if you don't want to implement it
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    CustomAllocator& impl = *static_cast<CustomAllocator*>(this_);
    ++impl.stats.num_allocs;
    impl.stats.max_alloc_size = std::max<int64_t>(size, impl.stats.max_alloc_size);

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

  static OrtStatus* ORT_API_CALL GetStatsImpl(const struct OrtAllocator* this_, OrtKeyValuePairs** out) noexcept {
    const CustomAllocator& impl = *static_cast<const CustomAllocator*>(this_);

    OrtKeyValuePairs* kvps;
    impl.api_ptrs.ort_api.CreateKeyValuePairs(&kvps);

    // if you wish to return stats the values in GetStatus should be formatted like this:
    // https://github.com/microsoft/onnxruntime/blob/2f878c60296de169a8a523e692d3d65893f7c133/onnxruntime/core/session/allocator_adapters.cc#L75-L85

    impl.api_ptrs.ort_api.AddKeyValuePair(kvps, "NumAllocs", std::to_string(impl.stats.num_allocs).c_str());
    impl.api_ptrs.ort_api.AddKeyValuePair(kvps, "MaxAllocSize", std::to_string(impl.stats.max_alloc_size).c_str());

    *out = kvps;
    return nullptr;
  }

 private:
  const OrtMemoryInfo* memory_info;
  const ApiPtrs api_ptrs;
  AllocatorStats stats{};
};
