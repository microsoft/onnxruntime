// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/tunable/util.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

using onnxruntime::rocm::CeilDiv;
using onnxruntime::rocm::aligned_vector;

namespace onnxruntime {

template <typename T, int VecSize>
std::string GenerateTritonKernelName() {
}

template <typename T, int ThreadsPerBlock, int VecSize>
Status LaunchVectorAdd(hipStream_t stream, const T* x, const T* y, T* z, int n) {
  hipLaunchKernelGGL((VectorAddKernel<T, VecSize>),
                     dim3(CeilDiv(n, ThreadsPerBlock*VecSize)),
                     dim3(ThreadsPerBlock),
                     0, stream,
                     x, y, z, n);
  auto status = hipGetLastError();
  ORT_RETURN_IF(status != hipSuccess, hipGetErrorName(status));
  return Status::OK();
}

}  // namespace onnxruntime
