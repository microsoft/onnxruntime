// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mkldnn_allocator.h"
#include "core/framework/allocatormgr.h"

namespace onnxruntime {

const OrtAllocatorInfo& MKLDNNAllocator::Info() const {
  static constexpr OrtAllocatorInfo mkl_allocator_info(MKLDNN, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
  return mkl_allocator_info;
}

const OrtAllocatorInfo& MKLDNNCPUAllocator::Info() const {
  static constexpr OrtAllocatorInfo mkl_cpu_allocator_info(MKLDNN_CPU, OrtAllocatorType::OrtDeviceAllocator, 0, ONNXRuntimeMemTypeCPUOutput);
  return mkl_cpu_allocator_info;
}
}  // namespace onnxruntime
