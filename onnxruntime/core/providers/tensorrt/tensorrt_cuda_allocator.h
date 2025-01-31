// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include "core/session/onnxruntime_c_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"

namespace onnxruntime {

// Following names are originally defined in allocator.h, in order to support out-of-tree/plugin EP framework,
// we shouldn't include allocator.h since it contains internal classes, ex: IAllocator. 
// Here simply duplicate the names.
constexpr const char* CUDA_ALLOCATOR = "Cuda";
constexpr const char* CUDA_PINNED_ALLOCATOR = "CudaPinned";

struct CUDAAllocator : OrtAllocator {
  CUDAAllocator(OrtDevice::DeviceId device_id, const char* name = onnxruntime::CUDA_ALLOCATOR) {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<CUDAAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<CUDAAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const CUDAAllocator*>(this_)->Info(); };
    mem_info_ = new OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                                  OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id),
                                  device_id, OrtMemTypeDefault);
  }
  //~CUDAAllocator();

  void* Alloc(size_t size);
  void Free(void* p);
  const OrtMemoryInfo* Info() const;

 private:
  CUDAAllocator(const CUDAAllocator&) = delete;
  CUDAAllocator& operator=(const CUDAAllocator&) = delete;

  void CheckDevice(bool throw_when_fail) const;
  void SetDevice(bool throw_when_fail) const;

  OrtMemoryInfo* mem_info_;
};

struct CUDAPinnedAllocator : OrtAllocator {
  CUDAPinnedAllocator(const char* name = onnxruntime::CUDA_PINNED_ALLOCATOR) {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<CUDAPinnedAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<CUDAPinnedAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const CUDAPinnedAllocator*>(this_)->Info(); };
    mem_info_ = new OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                                  OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, 0 /*CPU device always with id 0*/),
                                  0, OrtMemTypeCPUOutput);
  }
  //~CUDAPinnedAllocator();

  void* Alloc(size_t size);
  void Free(void* p);
  const OrtMemoryInfo* Info() const;

 private:
  CUDAPinnedAllocator(const CUDAPinnedAllocator&) = delete;
  CUDAPinnedAllocator& operator=(const CUDAPinnedAllocator&) = delete;

  OrtMemoryInfo* mem_info_;
};


}  // namespace onnxruntime
