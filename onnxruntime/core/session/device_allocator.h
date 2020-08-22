// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/framework/allocator.h"
#include "core/framework/utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_apis.h"
#include <assert.h>
#include "core/framework/allocatormgr.h"

namespace onnxruntime {
struct OrtAllocatorForDevice : public OrtAllocator {
  explicit OrtAllocatorForDevice(onnxruntime::AllocatorPtr&& dev_allocator)
      : device_allocator_(std::move(dev_allocator)) {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<OrtAllocatorForDevice*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<OrtAllocatorForDevice*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const OrtAllocatorForDevice*>(this_)->Info(); };
  }

  ~OrtAllocatorForDevice() = default;

  void* Alloc(size_t size) const {
    return device_allocator_->Alloc(size);
  }
  void Free(void* p) const {
    device_allocator_->Free(p);
  }

  const OrtMemoryInfo* Info() const {
    return &device_allocator_->Info();
  }

  OrtAllocatorForDevice(const OrtAllocatorForDevice&) = delete;
  OrtAllocatorForDevice& operator=(const OrtAllocatorForDevice&) = delete;

  onnxruntime::IAllocator* GetAllocator() {
    return device_allocator_.get();
  }

 private:
  onnxruntime::AllocatorPtr device_allocator_;
};
}  // namespace onnxruntime
