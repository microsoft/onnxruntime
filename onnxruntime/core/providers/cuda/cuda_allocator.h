// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

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
  FencePtr CreateFence(const SessionState* session_state) override;

 private:
  void CheckDevice(bool throw_when_fail) const;
};

//TODO: add a default constructor
class CUDAPinnedAllocator : public IAllocator {
 public:
  CUDAPinnedAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, device_id),
                          device_id, OrtMemTypeCPUOutput)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  FencePtr CreateFence(const SessionState* session_state) override;
};

class TorchCUDAAllocator : public IAllocator {
 public:
  TorchCUDAAllocator(OrtDevice::DeviceId device_id, const char* name);
  void* Alloc(size_t size) override;
  void Free(void* p) override;
  // FencePtr CreateFence(const SessionState* session_state) override;

 private:
  // void CheckDevice(bool throw_when_fail) const;
  void* libtorch_;
  void* (*torchMalloc)(size_t);  // torch's alloc function handle
  void (*torchFree)(void*);      // torch's free function handle
  void (*torchEmptyCache)();      // torch's free function handle
  };

}  // namespace onnxruntime
