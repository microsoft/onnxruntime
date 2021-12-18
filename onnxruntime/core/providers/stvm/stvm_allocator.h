// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STVM_ALLOCATOR
#define STVM_ALLOCATOR

#include "core/framework/allocator.h"
#include "stvm_common.h"

namespace onnxruntime {

#define STVM_ALLOC_ALIGN 128

class STVMAllocator : public IAllocator {
 public:
   STVMAllocator() : STVMAllocator(OrtMemoryInfo("TVM",
                                                 OrtAllocatorType::OrtDeviceAllocator,
                                                 OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0),
                                                 0,
                                                 OrtMemTypeDefault)) {}
  explicit STVMAllocator(const OrtMemoryInfo& info)
    : IAllocator(info) {
      switch (info.device.Type()) {
      case OrtDevice::CPU:
          ctx = {kDLCPU, info.device.Id()};
          break;
      case OrtDevice::GPU:
          ctx = {kDLVulkan, info.device.Id()};
          break;
      default:
          ORT_NOT_IMPLEMENTED("Unsupported device");
          break;
      }
    }

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  DLDevice ctx;
};

}  // namespace onnxruntime
#endif // STVM_ALLOCATOR
