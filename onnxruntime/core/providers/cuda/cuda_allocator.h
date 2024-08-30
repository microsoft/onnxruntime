// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

class CUDAAllocator : public IAllocator {
 public:
  CUDAAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id),
                          device_id, OrtMemTypeDefault)) {}
  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  void CheckDevice(bool throw_when_fail) const;
  void SetDevice(bool throw_when_fail) const;
};

class CUDAExternalAllocator : public CUDAAllocator {
  typedef void* (*ExternalAlloc)(size_t size);
  typedef void (*ExternalFree)(void* p);
  typedef void (*ExternalEmptyCache)();

 public:
  CUDAExternalAllocator(OrtDevice::DeviceId device_id, const char* name, void* alloc, void* free, void* empty_cache)
      : CUDAAllocator(device_id, name) {
    alloc_ = reinterpret_cast<ExternalAlloc>(alloc);
    free_ = reinterpret_cast<ExternalFree>(free);
    empty_cache_ = reinterpret_cast<ExternalEmptyCache>(empty_cache);
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  void* Reserve(size_t size) override;

 private:
  mutable OrtMutex lock_;
  ExternalAlloc alloc_;
  ExternalFree free_;
  ExternalEmptyCache empty_cache_;
  InlinedHashSet<void*> reserved_;
};

// TODO: add a default constructor
class CUDAPinnedAllocator : public IAllocator {
 public:
  CUDAPinnedAllocator(const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, 0 /*CPU device always with id 0*/),
                          0, OrtMemTypeCPUOutput)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};
}  // namespace onnxruntime
