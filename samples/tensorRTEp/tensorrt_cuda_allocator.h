// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include "core/session/onnxruntime_c_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {

// Following names are originally defined in allocator.h
constexpr const char* CUDA_ALLOCATOR = "Cuda";
constexpr const char* CUDA_PINNED_ALLOCATOR = "CudaPinned";

using DeviceId = int16_t;

struct CUDAAllocator : OrtAllocator {
  CUDAAllocator(DeviceId device_id, const char* name = onnxruntime::CUDA_ALLOCATOR) {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<CUDAAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<CUDAAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const CUDAAllocator*>(this_)->Info(); };

    device_id_ = device_id;

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    api->CreateMemoryInfo(name,
                          OrtAllocatorType::OrtDeviceAllocator,
                          static_cast<int>(device_id),
                          OrtMemType::OrtMemTypeDefault,
                          &mem_info_);
  }
  //~CUDAAllocator();

  void* Alloc(size_t size);
  void Free(void* p);
  const OrtMemoryInfo* Info() const;
  DeviceId GetDeviceId() const { return device_id_; };

 private:
  CUDAAllocator(const CUDAAllocator&) = delete;
  CUDAAllocator& operator=(const CUDAAllocator&) = delete;

  void CheckDevice(bool throw_when_fail) const;
  void SetDevice(bool throw_when_fail) const;

  DeviceId device_id_;
  OrtMemoryInfo* mem_info_ = nullptr;
};

struct CUDAPinnedAllocator : OrtAllocator {
  CUDAPinnedAllocator(const char* name = onnxruntime::CUDA_PINNED_ALLOCATOR) {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<CUDAPinnedAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<CUDAPinnedAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const CUDAPinnedAllocator*>(this_)->Info(); };
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    api->CreateMemoryInfo(name,
                          OrtAllocatorType::OrtDeviceAllocator,
                          0 /* CPU device always with id 0 */,
                          OrtMemType::OrtMemTypeDefault,
                          &mem_info_);
  }
  //~CUDAPinnedAllocator();

  void* Alloc(size_t size);
  void Free(void* p);
  const OrtMemoryInfo* Info() const;

  DeviceId GetDeviceId() const { return device_id_; };

 private:
  CUDAPinnedAllocator(const CUDAPinnedAllocator&) = delete;
  CUDAPinnedAllocator& operator=(const CUDAPinnedAllocator&) = delete;

  DeviceId device_id_ = 0;
  OrtMemoryInfo* mem_info_ = nullptr;
};


}  // namespace onnxruntime
