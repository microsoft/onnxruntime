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

ONNXRUNTIME_ALLOCATOR_IMPL_BEGIN(ONNXRuntimeDefaultAllocator)
private:
ONNXRuntimeAllocatorInfo* cpuAllocatorInfo;
ONNXRuntimeDefaultAllocator() : ref_count_(1){
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateAllocatorInfo("Cpu", ONNXRuntimeDeviceAllocator, 0, ONNXRuntimeMemTypeDefault, &cpuAllocatorInfo));
}
~ONNXRuntimeDefaultAllocator() {
  assert(ref_count_ == 0);
  ReleaseONNXRuntimeAllocatorInfo(cpuAllocatorInfo);
}

public:
ONNXRuntimeDefaultAllocator(const ONNXRuntimeDefaultAllocator&) = delete;
ONNXRuntimeDefaultAllocator& operator=(const ONNXRuntimeDefaultAllocator&) = delete;
ONNXRuntimeAllocatorInteface** Upcast() {
  return const_cast<ONNXRuntimeAllocatorInteface**>(&vtable_);
}
static ONNXRuntimeAllocatorInteface** Create() {
  return (ONNXRuntimeAllocatorInteface**)new ONNXRuntimeDefaultAllocator();
}
void* Alloc(size_t size) {
  return ::malloc(size);
}
void Free(void* p) {  
  return ::free(p);
}
const ONNXRuntimeAllocatorInfo* Info() {
  return cpuAllocatorInfo;
}
ONNXRUNTIME_ALLOCATOR_IMPL_END

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                                   \
  }                                                                    \
  catch (std::exception & ex) {                                        \
    return CreateONNXStatus(ONNXRUNTIME_RUNTIME_EXCEPTION, ex.what()); \
  }

ONNXRuntimeAllocatorInteface ONNXRuntimeDefaultAllocator::table_ = {
    {ONNXRuntimeDefaultAllocator::AddRef_, ONNXRuntimeDefaultAllocator::Release_}, ONNXRuntimeDefaultAllocator::Alloc_, ONNXRuntimeDefaultAllocator::Free_, ONNXRuntimeDefaultAllocator::Info_};

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateDefaultAllocator, _Out_ ONNXRuntimeAllocator** out){
    API_IMPL_BEGIN
    *out = ONNXRuntimeDefaultAllocator::Create();
    return nullptr;
    API_IMPL_END
}