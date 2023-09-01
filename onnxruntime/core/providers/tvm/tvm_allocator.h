// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_ALLOCATOR
#define TVM_ALLOCATOR

#include "core/framework/allocator.h"
#include "tvm_common.h"

namespace onnxruntime {
namespace tvm {

#define TVM_ALLOC_ALIGN 128

class TVMAllocator : public IAllocator {
 public:
  TVMAllocator() : TVMAllocator(OrtMemoryInfo("TVM",
                                              OrtAllocatorType::OrtDeviceAllocator,
                                              OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0),
                                              0,
                                              OrtMemTypeDefault)) {}
  explicit TVMAllocator(const OrtMemoryInfo& info)
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

}  // namespace tvm
}  // namespace onnxruntime

#endif  // TVM_ALLOCATOR
