// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/heavily_compressed_attention.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ __forceinline__ float ToFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float ToFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T FromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half FromFloat<half>(float value) {
  return __float2half(value);
}

template <typename T>
__global__ void HcaBlockBiasKernel(T* output, const int64_t* position_ids,
                                   int batch_size, int sequence_length, int entry_count, int compress_rate) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = batch_size * sequence_length * entry_count;
  if (index >= count) return;
  const int entry = index % entry_count;
  const int query = (index / entry_count) % sequence_length;
  const int batch = index / (sequence_length * entry_count);
  const int visible = min(entry_count,
                          static_cast<int>((position_ids[batch * sequence_length + query] + 1) / compress_rate));
  output[index] = FromFloat<T>(entry < visible ? 0.0f : -INFINITY);
}

template <typename T>
Status LaunchHcaBlockBiasKernel(cudaStream_t stream, T* block_bias, const int64_t* position_ids,
                                int batch_size, int sequence_length, int entry_count,
                                int compress_rate, int max_threads_per_block) {
  if (entry_count == 0) return Status::OK();
  const int count = batch_size * sequence_length * entry_count;
  const int block = std::min(max_threads_per_block, 256);
    HcaBlockBiasKernel<T><<<(count + block - 1) / block, block, 0, stream>>>(
      block_bias, position_ids, batch_size, sequence_length, entry_count, compress_rate);
  return CUDA_CALL(cudaGetLastError());
}


template Status LaunchHcaBlockBiasKernel<half>(cudaStream_t, half*, const int64_t*, int, int, int, int, int); template Status LaunchHcaBlockBiasKernel<BFloat16>(cudaStream_t, BFloat16*, const int64_t*, int, int, int, int, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
