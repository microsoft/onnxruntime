// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/reduction/all.h"

#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
namespace onnxruntime {
namespace cuda {

__global__ void assign_true(bool* ptr) {
  *ptr = true;
}

__global__ void assign_false(bool* ptr) {
  *ptr = false;
}

template<>
void LaunchAllKernel(cudaStream_t stream, const bool* data, const int size, bool* output) {
  if(thrust::all_of(thrust::device, data, data + size, thrust::identity<bool>())) {
    assign_true<<<1, 1, 0, stream>>>(output);
  }
  else
  {
    assign_false<<<1, 1, 0, stream>>>(output);
  }
}

}  // namespace cuda
}  // namespace onnxruntime
