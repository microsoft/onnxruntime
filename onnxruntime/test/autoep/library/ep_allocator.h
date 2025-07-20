// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "example_plugin_ep_utils.h"

#include <sstream>

// from onnxruntime/core/framework/allocator_stats.h
// copied from onnxruntime::AllocatorStats
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
                                  // is known. Certain allocator may return 0 to indicate the limit is unknown.
  int64_t bytes_limit;

  void ToKeyValuePairs(const OrtApi& api, OrtKeyValuePairs* kvps) const {
    if (num_allocs > 0 || bytes_limit != 0) {
      api.AddKeyValuePair(kvps, "Limit", std::to_string(bytes_limit).c_str());
      api.AddKeyValuePair(kvps, "InUse", std::to_string(bytes_in_use).c_str());
      api.AddKeyValuePair(kvps, "TotalAllocated", std::to_string(total_allocated_bytes).c_str());
      api.AddKeyValuePair(kvps, "MaxInUse", std::to_string(max_bytes_in_use).c_str());
      api.AddKeyValuePair(kvps, "NumAllocs", std::to_string(num_allocs).c_str());
      api.AddKeyValuePair(kvps, "NumReserves", std::to_string(num_reserves).c_str());
      api.AddKeyValuePair(kvps, "NumArenaExtensions", std::to_string(num_arena_extensions).c_str());
      api.AddKeyValuePair(kvps, "NumArenaShrinkages", std::to_string(num_arena_shrinkages).c_str());
      api.AddKeyValuePair(kvps, "MaxAllocSize", std::to_string(max_alloc_size).c_str());
    }
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
