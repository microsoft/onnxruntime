// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <mutex>

#include <gsl/gsl>

#include "core/framework/allocator.h"
#include "core/framework/plugin_ep_stream.h"

namespace onnxruntime {
namespace ep {
namespace adapter {

// Wraps an OrtAllocator* exposed by the C API as an IAllocator.
// Takes ownership of the OrtAllocator* and releases it on destruction.
class IAllocatorWrappingOrtAllocator final : public IAllocator {
 public:
  explicit IAllocatorWrappingOrtAllocator(gsl::not_null<OrtAllocator*> ort_allocator)
      : IAllocator(*ort_allocator->Info(ort_allocator)), ort_allocator_(ort_allocator.get()) {}

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
    return SupportsAllocOnStream();
  }

  void* AllocOnStream(size_t size, Stream* stream) override {
    // TODO: Replace direct function pointer access once Ort::Allocator exposes AllocOnStream/IsStreamAware.
    if (SupportsAllocOnStream()) {
      OrtAllocator* raw = ort_allocator_;
      return raw->AllocOnStream(raw, size, static_cast<OrtSyncStream*>(stream));
    }
    return Alloc(size);
  }

 private:
  static constexpr uint32_t kOrtAllocatorAllocOnStreamMinVersion = 23;

  bool SupportsAllocOnStream() const {
    const OrtAllocator* raw = ort_allocator_;
    return raw->version >= kOrtAllocatorAllocOnStreamMinVersion && raw->AllocOnStream != nullptr;
  }

  // TODO: Consider adding GetStats() override. Requires parsing OrtKeyValuePairs from the C API
  // into AllocatorStats; see GetStatsFromOrtAllocator() in allocator_adapters.cc for reference.

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
