// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator.h"
#include "core/framework/utils.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_apis.h"
#include <assert.h>

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

 private:
  onnxruntime::AllocatorPtr device_allocator_;
};

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                                \
  }                                                                 \
  catch (const std::exception& ex) {                                     \
    return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what()); \
  }

ORT_API_STATUS_IMPL(OrtApis::CreateAllocator, const OrtSession* sess, const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  auto allocator_ptr = session->GetAllocator(*mem_info);
  if (!allocator_ptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
  }
  *out = new OrtAllocatorForDevice(std::move(allocator_ptr));
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseAllocator, _Frees_ptr_opt_ OrtAllocator* allocator) {
  delete reinterpret_cast<OrtAllocatorForDevice*>(allocator);
}
