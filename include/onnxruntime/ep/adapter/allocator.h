// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#include "core/common/parse_string.h"
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
        owned_ort_allocator_(std::move(ort_allocator)),
        ort_allocator_(owned_ort_allocator_) {
  }

  // Wraps an OrtAllocator without taking ownership. The caller must keep it alive for the
  // lifetime of this IAllocator. This is used for allocators passed to PrePackWeight, which are
  // provided and owned by the ORT framework/caller for the duration of the PrePack call (this
  // wrapper must not release them).
  explicit IAllocatorWrappingOrtAllocator(OrtAllocator* ort_allocator)
      : IAllocator(*EnsureOrtAllocatorHasValue(ort_allocator)->Info(ort_allocator)),
        owned_ort_allocator_(nullptr),
        ort_allocator_(ort_allocator) {
  }

  void* Alloc(size_t size) override {
    return ort_allocator_->Alloc(ort_allocator_, size);
  }

  void Free(void* p) override {
    ort_allocator_->Free(ort_allocator_, p);
  }

  void* Reserve(size_t size) override {
    if (ort_allocator_->version >= 18 && ort_allocator_->Reserve != nullptr) {
      return ort_allocator_->Reserve(ort_allocator_, size);
    }
    return ort_allocator_->Alloc(ort_allocator_, size);
  }

  bool IsStreamAware() const override {
    static constexpr uint32_t kOrtAllocatorAllocOnStreamMinVersion = 23;
    return ort_allocator_->version >= kOrtAllocatorAllocOnStreamMinVersion && ort_allocator_->AllocOnStream != nullptr;
  }

  void* AllocOnStream(size_t size, Stream* stream) override {
    static constexpr uint32_t kOrtAllocatorAllocOnStreamMinVersion = 23;
    if (ort_allocator_->version >= kOrtAllocatorAllocOnStreamMinVersion && ort_allocator_->AllocOnStream != nullptr) {
      return ort_allocator_->AllocOnStream(ort_allocator_, size, reinterpret_cast<OrtSyncStream*>(stream));
    }

    return ort_allocator_->Alloc(ort_allocator_, size);
  }

  void GetStats(AllocatorStats* stats) override {
    if (!stats) return;
    *stats = {};

    // GetStats was added in OrtAllocator version 23. For older allocators the function pointer
    // may be uninitialized, so we must not call through it.
    if (ort_allocator_->version < 23 || !ort_allocator_->GetStats) return;

    OrtKeyValuePairs* stats_kvps = nullptr;
    Ort::ThrowOnError(ort_allocator_->GetStats(ort_allocator_, &stats_kvps));
    Ort::KeyValuePairs kvps{stats_kvps};
    std::vector<const char*> keys, values;
    kvps.GetKeyValuePairs(keys, values);
    const size_t n = keys.size() < values.size() ? keys.size() : values.size();
    for (size_t i = 0; i < n; ++i) {
      int64_t val = 0;
      if (!TryParseStringWithClassicLocale(std::string_view(values[i]), val)) continue;
      stats->SetFromKeyValue(keys[i], val);
    }
  }

 private:
  static const Ort::Allocator& EnsureOrtAllocatorHasValue(const Ort::Allocator& ort_allocator) {
    ORT_ENFORCE(ort_allocator != nullptr, "Ort::Allocator must contain a non-nullptr OrtAllocator.");
    return ort_allocator;
  }

  static OrtAllocator* EnsureOrtAllocatorHasValue(OrtAllocator* ort_allocator) {
    ORT_ENFORCE(ort_allocator != nullptr, "OrtAllocator must be non-null.");
    return ort_allocator;
  }

  Ort::Allocator owned_ort_allocator_;
  OrtAllocator* ort_allocator_{};

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
