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
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0),
                          0, OrtMemTypeDefault)),
        context_{context} {
  }

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;
  void SetAllocHint(AllocHint hint) override {
      alloc_hint_ = hint;
  }

 private:
  AllocatorStats stats_;
  const WebGpuContext& context_;
  AllocHint alloc_hint_ = AllocHint::None;
};

}  // namespace webgpu
}  // namespace onnxruntime
