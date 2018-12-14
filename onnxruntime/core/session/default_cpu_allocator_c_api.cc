// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include "core/session/onnxruntime_cxx_api.h"
#include <assert.h>

#define ORT_ALLOCATOR_IMPL_BEGIN(CLASS_NAME)                                  \
  class CLASS_NAME {                                                          \
   private:                                                                   \
    const OrtAllocatorInterface* vtable_ = &table_;                           \
    std::atomic_int ref_count_;                                               \
    static void* ORT_API_CALL Alloc_(void* this_ptr, size_t size) {           \
      return ((CLASS_NAME*)this_ptr)->Alloc(size);                            \
    }                                                                         \
    static void ORT_API_CALL Free_(void* this_ptr, void* p) {                 \
      return ((CLASS_NAME*)this_ptr)->Free(p);                                \
    }                                                                         \
    static const OrtAllocatorInfo* ORT_API_CALL Info_(const void* this_ptr) { \
      return ((const CLASS_NAME*)this_ptr)->Info();                           \
    }                                                                         \
    static uint32_t ORT_API_CALL AddRef_(void* this_) {                       \
      CLASS_NAME* this_ptr = (CLASS_NAME*)this_;                              \
      return ++this_ptr->ref_count_;                                          \
    }                                                                         \
    static uint32_t ORT_API_CALL Release_(void* this_) {                      \
      CLASS_NAME* this_ptr = (CLASS_NAME*)this_;                              \
      uint32_t ret = --this_ptr->ref_count_;                                  \
      if (ret == 0)                                                           \
        delete this_ptr;                                                      \
      return 0;                                                               \
    }                                                                         \
    static OrtAllocatorInterface table_;

#define ORT_ALLOCATOR_IMPL_END \
  }                            \
  ;

ORT_ALLOCATOR_IMPL_BEGIN(OrtDefaultAllocator)
private:
OrtAllocatorInfo* cpuAllocatorInfo;
OrtDefaultAllocator() : ref_count_(1) {
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cpuAllocatorInfo));
}
~OrtDefaultAllocator() {
  assert(ref_count_ == 0);
  ReleaseOrtAllocatorInfo(cpuAllocatorInfo);
}

public:
OrtDefaultAllocator(const OrtDefaultAllocator&) = delete;
OrtDefaultAllocator& operator=(const OrtDefaultAllocator&) = delete;
OrtAllocatorInterface** Upcast() {
  return const_cast<OrtAllocatorInterface**>(&vtable_);
}
static OrtAllocatorInterface** Create() {
  return (OrtAllocatorInterface**)new OrtDefaultAllocator();
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
ORT_ALLOCATOR_IMPL_END

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                           \
  }                                                            \
  catch (std::exception & ex) {                                \
    return CreateONNXStatus(ORT_RUNTIME_EXCEPTION, ex.what()); \
  }

OrtAllocatorInterface OrtDefaultAllocator::table_ = {
    {OrtDefaultAllocator::AddRef_, OrtDefaultAllocator::Release_}, OrtDefaultAllocator::Alloc_, OrtDefaultAllocator::Free_, OrtDefaultAllocator::Info_};

ORT_API_STATUS_IMPL(OrtCreateDefaultAllocator, _Out_ OrtAllocator** out) {
  API_IMPL_BEGIN
  *out = OrtDefaultAllocator::Create();
  return nullptr;
  API_IMPL_END
}
