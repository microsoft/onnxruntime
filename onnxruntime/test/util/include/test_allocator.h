// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/allocator.h"
#include <atomic>
#include <stdexcept>
#include "core/framework/allocator_info.h"
#include "core/session/onnxruntime_cxx_api.h"
#include <assert.h>

#define ONNXRUNTIME_ALLOCATOR_IMPL_BEGIN(CLASS_NAME)                                          \
  class CLASS_NAME {                                                                          \
   private:                                                                                   \
    const ONNXRuntimeAllocatorInteface* vtable_ = &table_;                                    \
    std::atomic_int ref_count_;                                                               \
    static void* ONNXRUNTIME_API_STATUSCALL Alloc_(void* this_ptr, size_t size) {             \
      return ((CLASS_NAME*)this_ptr)->Alloc(size);                                            \
    }                                                                                         \
    static void ONNXRUNTIME_API_STATUSCALL Free_(void* this_ptr, void* p) {                   \
      return ((CLASS_NAME*)this_ptr)->Free(p);                                                \
    }                                                                                         \
    static const ONNXRuntimeAllocatorInfo* ONNXRUNTIME_API_STATUSCALL Info_(void* this_ptr) { \
      return ((CLASS_NAME*)this_ptr)->Info();                                                 \
    }                                                                                         \
    static uint32_t ONNXRUNTIME_API_STATUSCALL AddRef_(void* this_) {                         \
      CLASS_NAME* this_ptr = (CLASS_NAME*)this_;                                              \
      return ++this_ptr->ref_count_;                                                          \
    }                                                                                         \
    static uint32_t ONNXRUNTIME_API_STATUSCALL Release_(void* this_) {                        \
      CLASS_NAME* this_ptr = (CLASS_NAME*)this_;                                              \
      uint32_t ret = --this_ptr->ref_count_;                                                  \
      if (ret == 0)                                                                           \
        delete this_ptr;                                                                      \
      return 0;                                                                               \
    }                                                                                         \
    static ONNXRuntimeAllocatorInteface table_;

#define ONNXRUNTIME_ALLOCATOR_IMPL_END \
  }                                    \
  ;

ONNXRUNTIME_ALLOCATOR_IMPL_BEGIN(MockedONNXRuntimeAllocator)
private:
std::atomic<size_t> memory_inuse;
ONNXRuntimeAllocatorInfo* cpuAllocatorInfo;
MockedONNXRuntimeAllocator() : ref_count_(1), memory_inuse(0) {
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateAllocatorInfo("Cpu", ONNXRuntimeDeviceAllocator, 0, ONNXRuntimeMemTypeDefault, &cpuAllocatorInfo));
}
~MockedONNXRuntimeAllocator() {
  assert(ref_count_ == 0);
  ReleaseONNXRuntimeAllocatorInfo(cpuAllocatorInfo);
}

public:
 MockedONNXRuntimeAllocator(const MockedONNXRuntimeAllocator&) = delete;
 MockedONNXRuntimeAllocator& operator=(const MockedONNXRuntimeAllocator&) = delete;
 ONNXRuntimeAllocatorInteface** Upcast() {
   return const_cast<ONNXRuntimeAllocatorInteface**>(&vtable_);
}
static ONNXRuntimeAllocatorInteface** Create() {
  return (ONNXRuntimeAllocatorInteface**)new MockedONNXRuntimeAllocator();
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
const ONNXRuntimeAllocatorInfo* Info() {
  return cpuAllocatorInfo;
}

void LeakCheck() {
  if (memory_inuse.load())
    throw std::runtime_error("memory leak!!!");
}
ONNXRUNTIME_ALLOCATOR_IMPL_END

