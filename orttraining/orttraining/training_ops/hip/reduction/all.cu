// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/hip/reduction/all.h"

#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace onnxruntime {
namespace hip {

__global__ void assign_true(bool* ptr) {
  *ptr = true;
}

__global__ void assign_false(bool* ptr) {
  *ptr = false;
}

template<>
void LaunchAllKernel(const bool* data, const int size, bool* output) {
  if(thrust::all_of(thrust::device, data, data + size, thrust::identity<bool>())) {
    hipLaunchKernelGGL(assign_true, dim3(1), dim3(1), 0, 0, output);
  }
  else
  {
    hipLaunchKernelGGL(assign_false, dim3(1), dim3(1), 0, 0, output);
  }
}

}  // namespace hip
}  // namespace onnxruntime
