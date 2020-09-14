// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

class CUDAAllocator : public Provider_IAllocator {
 public:
  CUDAAllocator(OrtDevice::DeviceId device_id, const char* name)
      : Provider_IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id),
                          device_id, OrtMemTypeDefault)) {}
  void* Alloc(size_t size) override;
  void Free(void* p) override;
  FencePtr CreateFence(const Provider_SessionState* session_state) override;

 private:
  void CheckDevice(bool throw_when_fail) const;
};

//TODO: add a default constructor
class CUDAPinnedAllocator : public Provider_IAllocator {
 public:
  CUDAPinnedAllocator(OrtDevice::DeviceId device_id, const char* name)
      : Provider_IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, device_id),
                          device_id, OrtMemTypeCPUOutput)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  FencePtr CreateFence(const Provider_SessionState* session_state) override;
};
}  // namespace onnxruntime
