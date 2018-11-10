// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mkldnn_allocator.h"
#include "core/framework/allocatormgr.h"

namespace onnxruntime {

const ONNXRuntimeAllocatorInfo& MKLDNNAllocator::Info() const {
  static constexpr ONNXRuntimeAllocatorInfo mkl_allocator_info(MKLDNN, ONNXRuntimeAllocatorType::ONNXRuntimeDeviceAllocator, 0, ONNXRuntimeMemTypeDefault);
  return mkl_allocator_info;
}

const ONNXRuntimeAllocatorInfo& MKLDNNCPUAllocator::Info() const {
  static constexpr ONNXRuntimeAllocatorInfo mkl_cpu_allocator_info(MKLDNN_CPU, ONNXRuntimeAllocatorType::ONNXRuntimeDeviceAllocator, 0, ONNXRuntimeMemTypeCPUOutput);
  return mkl_cpu_allocator_info;
}
}  // namespace onnxruntime
