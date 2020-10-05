// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/allocator.h"
#include "core/framework/arena.h"
#include "core/session/device_allocator.h"

namespace onnxruntime {
class AllocatorWrapper : public IAllocator {
 public:
  AllocatorWrapper(OrtAllocator* impl) : IAllocator(*impl->Info(impl)), impl_(impl) {}
  void* Alloc(size_t size) override {
    return impl_->Alloc(impl_, size);
  }
  void Free(void* p) override {
    return impl_->Free(impl_, p);
  }

 private:
  OrtAllocator* impl_;
};

}  // namespace onnxruntime
