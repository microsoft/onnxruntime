// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include "core/session/onnxruntime_cxx_api.h"
#include <assert.h>

// In the future we'll have more than one allocator type. Since all allocators are of type 'OrtAllocator' and there is a single
// OrtReleaseAllocator function, we need to have a common base type that lets us delete them.
struct OrtAllocatorImpl : OrtAllocator {
  virtual ~OrtAllocatorImpl() = default;
};

struct OrtDefaultAllocator : OrtAllocatorImpl {
  OrtDefaultAllocator() {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<OrtDefaultAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<OrtDefaultAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const OrtDefaultAllocator*>(this_)->Info(); };
    ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cpuAllocatorInfo));
  }

  ~OrtDefaultAllocator() override { OrtReleaseAllocatorInfo(cpuAllocatorInfo); }

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

ORT_API(void, OrtReleaseAllocator, _In_ OrtAllocator* allocator) {
  delete static_cast<OrtAllocatorImpl*>(allocator);
}
