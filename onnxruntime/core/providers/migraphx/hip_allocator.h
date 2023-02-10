// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>
#include "core/framework/allocator.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

class HIPAllocator : public IAllocator {
 public:
  HIPAllocator(int device_id, const char* name)
    : IAllocator(
        OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                      OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id),
                      device_id, OrtMemTypeDefault)) {}

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;

 private:
  void CheckDevice() const;
};

class HIPExternalAllocator : public HIPAllocator {
  typedef void* (*ExternalAlloc)(size_t size);
  typedef void (*ExternalFree)(void* p);
  typedef void (*ExternalEmptyCache)();

 public:
  HIPExternalAllocator(OrtDevice::DeviceId device_id, const char* name, void* alloc, void* free, void* empty_cache)
      : HIPAllocator(device_id, name) {
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
  std::unordered_set<void*> reserved_;
};

//TODO: add a default constructor
class HIPPinnedAllocator : public IAllocator {
 public:
  HIPPinnedAllocator(int device_id, const char* name)
    : IAllocator(
          OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                        OrtDevice(OrtDevice::CPU, OrtDevice::MemType::HIP_PINNED, device_id),
                        device_id, OrtMemTypeCPUOutput)) {}

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
};

}  // namespace onnxruntime
