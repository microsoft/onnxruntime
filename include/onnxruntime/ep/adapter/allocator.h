// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "core/framework/allocator.h"

namespace onnxruntime {
namespace ep {
namespace adapter {

// Wraps an OrtAllocator* exposed by the C API as an IAllocator.
// Takes ownership of the wrapped Ort::Allocator and releases it on destruction.
class IAllocatorWrappingOrtAllocator final : public IAllocator {
 public:
  explicit IAllocatorWrappingOrtAllocator(Ort::Allocator ort_allocator)
      : IAllocator(*(EnsureOrtAllocatorHasValue(ort_allocator).GetInfo())),
        ort_allocator_(std::move(ort_allocator)) {
  }

  void* Alloc(size_t size) override {
    return ort_allocator_.Alloc(size);
  }

  void Free(void* p) override {
    ort_allocator_.Free(p);
  }

  void* Reserve(size_t size) override {
    return ort_allocator_.Reserve(size);
  }

  bool IsStreamAware() const override {
    return false;

    // TODO: Enable once AllocOnStream() is implemented.
    // static constexpr uint32_t kOrtAllocatorAllocOnStreamMinVersion = 23;
    // const OrtAllocator* raw = ort_allocator_;
    // return raw->version >= kOrtAllocatorAllocOnStreamMinVersion && raw->AllocOnStream != nullptr;
  }

  void* AllocOnStream(size_t /*size*/, Stream* /*stream*/) override {
    // TODO: Implement AllocOnStream().
    // The internal `onnxruntime::IAllocator::AllocOnStream` signature takes an internal `onnxruntime::Stream*`
    // argument, while the public `::OrtAllocator::AllocOnStream` signature takes an `::OrtSyncStream*` argument.
    // We need to properly map from one to the other.
    // `::OrtSyncStream*` should be treated as an opaque type from the plugin EP's perspective.
    ORT_NOT_IMPLEMENTED("IAllocatorWrappingOrtAllocator::AllocOnStream is not implemented yet.");
  }

  void GetStats(AllocatorStats* stats) override {
    if (!stats) return;
    *stats = {};

    try {
      Ort::KeyValuePairs kvps = ort_allocator_.GetStats();
      std::vector<const char*> keys, values;
      kvps.GetKeyValuePairs(keys, values);
      for (size_t i = 0; i < keys.size(); ++i) {
        char* end = nullptr;
        int64_t val = std::strtoll(values[i], &end, 10);
        if (end == values[i]) continue;  // skip unparseable entries
        if (std::strcmp(keys[i], "Limit") == 0) {
          stats->bytes_limit = val;
        } else if (std::strcmp(keys[i], "InUse") == 0) {
          stats->bytes_in_use = val;
        } else if (std::strcmp(keys[i], "RequestedInUse") == 0) {
          stats->bytes_requested_in_use = val;
        } else if (std::strcmp(keys[i], "TotalAllocated") == 0) {
          stats->total_allocated_bytes = val;
        } else if (std::strcmp(keys[i], "MaxInUse") == 0) {
          stats->max_bytes_in_use = val;
        } else if (std::strcmp(keys[i], "NumAllocs") == 0) {
          stats->num_allocs = val;
        } else if (std::strcmp(keys[i], "NumReserves") == 0) {
          stats->num_reserves = val;
        } else if (std::strcmp(keys[i], "NumArenaExtensions") == 0) {
          stats->num_arena_extensions = val;
        } else if (std::strcmp(keys[i], "NumArenaShrinkages") == 0) {
          stats->num_arena_shrinkages = val;
        } else if (std::strcmp(keys[i], "MaxAllocSize") == 0) {
          stats->max_alloc_size = val;
        }
      }
    } catch (...) {
      // If plugin doesn't implement GetStats, AllocatorGetStats returns empty KVPs.
      // Any other failure is silently ignored.
    }
  }

 private:
  static const Ort::Allocator& EnsureOrtAllocatorHasValue(const Ort::Allocator& ort_allocator) {
    ORT_ENFORCE(ort_allocator != nullptr, "Ort::Allocator must contain a non-nullptr OrtAllocator.");
    return ort_allocator;
  }

  Ort::Allocator ort_allocator_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(IAllocatorWrappingOrtAllocator);
};

/// <summary>
/// A bridge class between the EP API OrtAllocator and an IAllocator implementation.
/// </summary>
class Allocator : public OrtAllocator {
 public:
  /**
   * Create from an existing AllocatorPtr.
   */
  explicit Allocator(const OrtMemoryInfo* memory_info, AllocatorPtr impl)
      : Allocator{memory_info} {
    ORT_ENFORCE(impl != nullptr, "Allocator implementation cannot be null.");
    impl_ = impl;
  }

  using AllocatorFactory = AllocatorPtr (*)(const OrtMemoryInfo& memory_info);

  /**
   * Create from an AllocatorFactory, which will be called lazily when the first allocation is made.
   */
  explicit Allocator(const OrtMemoryInfo* memory_info, AllocatorFactory get_allocator_impl)
      : Allocator{memory_info} {
    get_allocator_impl_ = get_allocator_impl;
  }

 private:
  explicit Allocator(const OrtMemoryInfo* memory_info)
      : OrtAllocator{}, memory_info_(memory_info), get_allocator_impl_(nullptr) {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
  }
  AllocatorPtr GetImpl() {
    if (!impl_) {
      std::call_once(init_flag_, [this]() {
        impl_ = get_allocator_impl_(*memory_info_);
      });
    }
    return impl_;
  }

  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept {
    auto* allocator = static_cast<Allocator*>(this_ptr);
    return allocator->GetImpl()->Alloc(size);
  }

  static void ORT_API_CALL FreeImpl(OrtAllocator* this_ptr, void* p) noexcept {
    auto* allocator = static_cast<Allocator*>(this_ptr);
    allocator->GetImpl()->Free(p);
  }

  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_ptr) noexcept {
    auto* allocator = static_cast<const Allocator*>(this_ptr);
    return allocator->memory_info_;
  }

  const OrtMemoryInfo* memory_info_;
  AllocatorPtr impl_;
  AllocatorFactory get_allocator_impl_;
  std::once_flag init_flag_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
