// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {
constexpr const char* TRT = "Trt";

class TensorrtPinnedAllocator : public CPUAllocator {
 public:
  virtual const OrtAllocatorInfo& Info() const override {
    static OrtAllocatorInfo tensorrt_cpu_allocator_info(TRT,
                                                   OrtAllocatorType::OrtDeviceAllocator, 0,
                                                   OrtMemType::OrtMemTypeCPU);
    return tensorrt_cpu_allocator_info;
  }
};

/*! \brief The default allocator doesn't allocate anything. It's used here to let allocation
           planner get allocator information.
*/
class TensorrtAllocator : public CPUAllocator {
 public:
  virtual const OrtAllocatorInfo& Info() const override {
    static OrtAllocatorInfo tensorrt_default_allocator_info(TRT,
                                                       OrtAllocatorType::OrtDeviceAllocator, 0,
                                                       OrtMemType::OrtMemTypeDefault);
    return tensorrt_default_allocator_info;
  }
};
}  // namespace onnxruntime
