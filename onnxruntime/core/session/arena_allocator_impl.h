// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/allocator.h"
#include "core/framework/arena.h"
#include "core/session/device_allocator.h"

namespace onnxruntime {
class ArenaAllocatorWrapper : public IArenaAllocator {
 public:
  ArenaAllocatorWrapper(OrtAllocatorArena* impl) : IArenaAllocator(*impl->device_allocator->Info(impl->device_allocator)),
                                                                            impl_(impl){}
  void* Alloc(size_t size) override {
    return impl_->Alloc(size);
  }
  void Free(void* p) override {
    return impl_->Free(p);
  }
  void* Reserve(size_t size) override {
    return impl_->Reserve(size);
  }
  size_t Used() const override {
    return impl_->Used();
  }
  size_t Max() const override {
    return impl_->Max();
  }

 private:
  OrtAllocatorArena* impl_;
};

}  // namespace onnxruntime