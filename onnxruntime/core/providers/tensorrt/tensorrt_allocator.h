// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {

constexpr const char* TRT = "Tensorrt";
constexpr const char* TRT_PINNED = "TensorrtPinned";

class TensorrtAllocator : public IDeviceAllocator {
 public:
  TensorrtAllocator(int device_id) : info_(TRT, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id), device_id, OrtMemTypeDefault) {}
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const OrtAllocatorInfo& Info() const override;
  virtual FencePtr CreateFence(const SessionState* session_state) override;

 private:
  void CheckDevice() const;

 private:
  const OrtAllocatorInfo info_;
};

class TensorrtPinnedAllocator : public IDeviceAllocator {
 public:
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const OrtAllocatorInfo& Info() const override;
  virtual FencePtr CreateFence(const SessionState* session_state) override;
};

}  // namespace onnxruntime
