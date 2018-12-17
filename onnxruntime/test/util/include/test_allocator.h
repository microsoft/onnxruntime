// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <stdexcept>
#include "core/session/onnxruntime_c_api.h"
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

ORT_ALLOCATOR_IMPL_BEGIN(MockedOrtAllocator)
private:
std::atomic<size_t> memory_inuse;
OrtAllocatorInfo* cpuAllocatorInfo;
MockedOrtAllocator() : ref_count_(1), memory_inuse(0) {
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cpuAllocatorInfo));
}
~MockedOrtAllocator() {
  assert(ref_count_ == 0);
  ReleaseOrtAllocatorInfo(cpuAllocatorInfo);
}

public:
MockedOrtAllocator(const MockedOrtAllocator&) = delete;
MockedOrtAllocator& operator=(const MockedOrtAllocator&) = delete;
OrtAllocatorInterface** Upcast() {
  return const_cast<OrtAllocatorInterface**>(&vtable_);
}
static OrtAllocatorInterface** Create() {
  return (OrtAllocatorInterface**)new MockedOrtAllocator();
}
void* Alloc(size_t size) {
  constexpr size_t extra_len = sizeof(size_t);
  memory_inuse.fetch_add(size += extra_len);
  void* p = ::malloc(size);
  *(size_t*)p = size;
  return (char*)p + extra_len;
}
void Free(void* p) {
  constexpr size_t extra_len = sizeof(size_t);
  if (!p) return;
  p = (char*)p - extra_len;
  size_t len = *(size_t*)p;
  memory_inuse.fetch_sub(len);
  return ::free(p);
}
const OrtAllocatorInfo* Info() const {
  return cpuAllocatorInfo;
}

void LeakCheck() {
  if (memory_inuse.load())
    throw std::runtime_error("memory leak!!!");
}
ORT_ALLOCATOR_IMPL_END
