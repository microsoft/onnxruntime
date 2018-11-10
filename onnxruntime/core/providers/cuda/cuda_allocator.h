// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {
constexpr const char* CUDA = "Cuda";
constexpr const char* CUDA_PINNED = "CudaPinned";

class CUDAAllocator : public IDeviceAllocator {
 public:
  CUDAAllocator(int device_id) : device_id_(device_id), info_(CUDA, ONNXRuntimeAllocatorType::ONNXRuntimeDeviceAllocator, device_id, ONNXRuntimeMemTypeDefault) {}
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const ONNXRuntimeAllocatorInfo& Info() const override;
  virtual FencePtr CreateFence(const SessionState* session_state) override;

 private:
  void CheckDevice() const;

 private:
  const int device_id_;
  const ONNXRuntimeAllocatorInfo info_;
};

//TODO: add a default constructor
class CUDAPinnedAllocator : public IDeviceAllocator {
 public:
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const ONNXRuntimeAllocatorInfo& Info() const override;
  virtual FencePtr CreateFence(const SessionState* session_state) override;
};

}  // namespace onnxruntime
