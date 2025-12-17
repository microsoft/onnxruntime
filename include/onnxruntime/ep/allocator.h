// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/pch.h instead."
#endif

#include "api.h"

#include "core/framework/allocator.h"

namespace onnxruntime {
namespace ep {

/// <summary>
/// A bridge class between the EP API OrtAllocator and an IAllocator implementation.
/// </summary>
class Allocator : public OrtAllocator {
 public:
  explicit Allocator(AllocatorPtr impl) : OrtAllocator{}, impl_(impl) {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
  }

 private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept {
    auto* allocator = static_cast<Allocator*>(this_ptr);
    return allocator->impl_->Alloc(size);
  }

  static void ORT_API_CALL FreeImpl(OrtAllocator* this_ptr, void* p) noexcept {
    auto* allocator = static_cast<Allocator*>(this_ptr);
    allocator->impl_->Free(p);
  }

  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_ptr) noexcept {
    auto* allocator = static_cast<const Allocator*>(this_ptr);
    return &allocator->impl_->Info();
  }

  AllocatorPtr impl_;
};

}  // namespace ep
}  // namespace onnxruntime
