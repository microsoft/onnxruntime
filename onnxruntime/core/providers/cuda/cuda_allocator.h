// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {

class CUDAAllocator : public IDeviceAllocator {
 public:
  CUDAAllocator(OrtDevice::DeviceId device_id, const char* name) : info_(name, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id), device_id, OrtMemTypeDefault) {}
  void* Alloc(size_t size) override;
  void Free(void* p) override;
  const OrtMemoryInfo& Info() const override;
  FencePtr CreateFence(const SessionState* session_state) override;

 private:
  void CheckDevice(bool throw_when_fail) const;

 private:
  const OrtMemoryInfo info_;
};

//TODO: add a default constructor
class CUDAPinnedAllocator : public IDeviceAllocator {
 public:
  CUDAPinnedAllocator(OrtDevice::DeviceId device_id, const char* name) : info_(name, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, device_id), device_id, OrtMemTypeCPUOutput) {}
  void* Alloc(size_t size) override;
  void Free(void* p) override;
  const OrtMemoryInfo& Info() const override;
  FencePtr CreateFence(const SessionState* session_state) override;

 private:
  const OrtMemoryInfo info_;
};

}  // namespace onnxruntime
