// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include "core/session/onnxruntime_cxx_api.h"
#include <assert.h>

struct OrtDefaultAllocator : OrtAllocator {
  OrtDefaultAllocator() {
    OrtAllocator::Alloc = [](void* this_, size_t size) { return reinterpret_cast<OrtDefaultAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](void* this_, void* p) { reinterpret_cast<OrtDefaultAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const void* this_) { return reinterpret_cast<const OrtDefaultAllocator*>(this_)->Info(); };
    ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cpuAllocatorInfo));
  }
  ~OrtDefaultAllocator() {
    OrtReleaseAllocatorInfo(cpuAllocatorInfo);
  }

  void* Alloc(size_t size) {
    return ::malloc(size);
  }
  void Free(void* p) {
    return ::free(p);
  }
  const OrtAllocatorInfo* Info() const {
    return cpuAllocatorInfo;
  }

 private:
  OrtDefaultAllocator(const OrtDefaultAllocator&) = delete;
  OrtDefaultAllocator& operator=(const OrtDefaultAllocator&) = delete;

  OrtAllocatorInfo* cpuAllocatorInfo;
};

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                          \
  }                                                           \
  catch (std::exception & ex) {                               \
    return OrtCreateStatus(ORT_RUNTIME_EXCEPTION, ex.what()); \
  }

ORT_API_STATUS_IMPL(OrtCreateDefaultAllocator, _Out_ OrtAllocator** out) {
  API_IMPL_BEGIN
  *out = new OrtDefaultAllocator();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtReleaseDefaultAllocator, _In_ OrtAllocator* allocator) {
  delete static_cast<OrtDefaultAllocator*>(allocator);
  return nullptr;
}
