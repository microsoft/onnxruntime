// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/gpt3_attention_impl.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void Gpt3AttentionKernel(const T* query, const T* key, const T* value, T* output) {

  const int element_count = gridDim.x;
  const int idx = blockIdx.x;

  if (idx < element_count) {
    output[idx] = (T)((float)query[idx] + (float)key[idx] + (float)value[idx]);
  }
}

template <typename T>
bool ComputeGpt3Attention(
  cudaStream_t stream,
  const int element_count,
  const T* query,
  const T* key,
  const T* value,
  T* output) {
  const dim3 grid(element_count, 1, 1);
  const dim3 block(1, 1, 1);

  Gpt3AttentionKernel<T><<<grid, block, 0, stream>>>(query, key, value, output);
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchGpt3AttentionKernel(
    cudaStream_t stream,
    void* output,
    const void* query,
    const void* key,
    const void* value,
    int element_count,
    size_t element_size) {
  if (element_size == 2) {
    return ComputeGpt3Attention(
        stream,
        element_count,
        reinterpret_cast<const half*>(query),
        reinterpret_cast<const half*>(key),
        reinterpret_cast<const half*>(value),
        reinterpret_cast<half*>(output));
  } else {
    return ComputeGpt3Attention(
        stream,
        element_count,
        reinterpret_cast<const float*>(query),
        reinterpret_cast<const float*>(key),
        reinterpret_cast<const float*>(value),
        reinterpret_cast<float*>(output));
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
