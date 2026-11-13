// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/common/scatter_mean.h"

namespace onnxruntime::cuda {

__device__ inline void LinkScatterUpdate(int64_t* heads, int64_t* next, int64_t destination, int64_t update) {
  next[update] = static_cast<int64_t>(
      atomicExch(reinterpret_cast<unsigned long long*>(heads + destination), static_cast<unsigned long long>(update)));
}

template <typename T>
__global__ void ScatterMeanKernel(const T* input, const T* updates, T* output,
                                  const int64_t* heads, const int64_t* next, int64_t size, int64_t slice_size) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < size; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t head = heads[i / slice_size];
    output[i] = head < 0 ? input[i] : scatter_internal::MeanOfLinkedUpdates(input[i], updates, next, head, slice_size, i % slice_size);
  }
}

}  // namespace onnxruntime::cuda
