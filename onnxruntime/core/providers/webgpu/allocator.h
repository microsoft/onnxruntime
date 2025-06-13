// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace webgpu {

class WebGpuContext;

class GpuBufferAllocator : public IAllocator {
 public:
  GpuBufferAllocator(const WebGpuContext& context)
      : IAllocator(
            OrtMemoryInfo(WEBGPU_BUFFER, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0),
                          OrtMemTypeDefault)),
        context_{context} {
  }

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

  void OnSessionInitializationStart(uint32_t session_id);
  void OnSessionInitializationEnd();

 private:
  AllocatorStats stats_;
  const WebGpuContext& context_;
  bool session_initialized_ = false;
  uint32_t session_id_ = 0;
};

}  // namespace webgpu
}  // namespace onnxruntime
