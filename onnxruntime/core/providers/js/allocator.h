// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace js {

class JsCPUInputAllocator : public CPUAllocator {
 public:
  JsCPUInputAllocator()
      : CPUAllocator(
            OrtMemoryInfo("JsCPUInputAllocator", OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0),
                          0, OrtMemTypeCPUInput)){};
};

class JsCPUOutputAllocator : public CPUAllocator {
 public:
  JsCPUOutputAllocator()
      : CPUAllocator(
            OrtMemoryInfo("JsCPUOutputAllocator", OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0),
                          0, OrtMemTypeCPUOutput)){};
};

class JsCustomAllocator : public IAllocator {
 public:
  JsCustomAllocator()
      : IAllocator(
            OrtMemoryInfo("JsCustomAllocator", OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HANDLE, 0),
                          0, OrtMemTypeDefault)) {
  }

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

 private:
  AllocatorStats stats_;
};

}  // namespace js
}  // namespace onnxruntime
