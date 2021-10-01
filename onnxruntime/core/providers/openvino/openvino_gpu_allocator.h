// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {

class OpenVINOAllocator : public IAllocator {
 public:
  OpenVINOAllocator()
      : IAllocator(
            OrtMemoryInfo("IGPU", OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,0),
                          0, OrtMemTypeDefault)) {}
  void* Alloc(size_t size) override;
  void Free(void* p) override;
};
}
