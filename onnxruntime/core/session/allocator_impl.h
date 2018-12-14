// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
class AllocatorWrapper : public IAllocator {
 public:
  AllocatorWrapper(OrtAllocator* impl) : impl_(impl) {
    (*impl)->parent.AddRef(impl);
  }
  ~AllocatorWrapper() {
    (*impl_)->parent.Release(impl_);
  }
  void* Alloc(size_t size) override {
    return (*impl_)->Alloc(impl_, size);
  }
  void Free(void* p) override {
    return (*impl_)->Free(impl_, p);
  }
  const OrtAllocatorInfo& Info() const override {
    return *(OrtAllocatorInfo*)(*impl_)->Info(impl_);
  }

 private:
  OrtAllocator* impl_;
};
}  // namespace onnxruntime
