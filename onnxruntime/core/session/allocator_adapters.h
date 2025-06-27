// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/session/onnxruntime_cxx_api.h"

#include <string>

namespace onnxruntime {

// Since all allocators are of type 'OrtAllocator' and there is a single
// OrtApis:ReleaseAllocator function, we need to have a common base type that lets us delete them.
struct OrtAllocatorImpl : public OrtAllocator {
  virtual ~OrtAllocatorImpl() = default;
};

// The following are "adapters" to allow using an IAllocator implementation wrapped as an OrtAllocator
// and vice versa to plug into any ORT internal code/ API implementation as necessary

struct OrtAllocatorImplWrappingIAllocator final : public OrtAllocatorImpl {
  explicit OrtAllocatorImplWrappingIAllocator(onnxruntime::AllocatorPtr&& i_allocator);

  ~OrtAllocatorImplWrappingIAllocator() override = default;

  void* Alloc(size_t size);
  void Free(void* p);
  void* Reserve(size_t size);

  const OrtMemoryInfo* Info() const;

  std::unordered_map<std::string, std::string> Stats() const;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtAllocatorImplWrappingIAllocator);

  onnxruntime::AllocatorPtr GetWrappedIAllocator();

 private:
  onnxruntime::AllocatorPtr i_allocator_;
};

using OrtAllocatorUniquePtr = std::unique_ptr<OrtAllocator, std::function<void(OrtAllocator*)>>;

class IAllocatorImplWrappingOrtAllocator final : public IAllocator {
 public:
  // ctor for OrtAllocator we do not own
  explicit IAllocatorImplWrappingOrtAllocator(OrtAllocator* ort_allocator);

  // ctor for OrtAllocator we own.
  explicit IAllocatorImplWrappingOrtAllocator(OrtAllocatorUniquePtr ort_allocator);

  // ~IAllocatorImplWrappingOrtAllocator() override = default;

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  void* Reserve(size_t size) override;

  const OrtAllocator* GetWrappedOrtAllocator() const {
    return ort_allocator_.get();
  }

  void GetStats(AllocatorStats* stats) override;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(IAllocatorImplWrappingOrtAllocator);

 private:
  OrtAllocatorUniquePtr ort_allocator_ = nullptr;
};

}  // namespace onnxruntime
