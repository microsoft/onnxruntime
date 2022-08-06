// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "add_bias_transpose.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void AddBiasTransposeQKV(const T* input, const T* biases, T* output) {
  // Input:  BxSxMxNxH  (Format 1)
  // Output: MxBxNxSxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int M = gridDim.z;
  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = n * head_size + (m + s * M) * NH + b * NHS * M;
  const int out_offset = s * head_size + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  const int h = threadIdx.x;
  if (h < head_size) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeQKVLarge(const int head_size, const T* input, const T* biases, T* output) {
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int stride = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int M = gridDim.z;
  const int H = head_size;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  int in_offset = n * H + (m + s * M) * NH + b * NHS * M;
  const int out_offset = s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}


template <typename T>
__global__ void AddBiasTranspose(const T* input, const T* biases, T* output) {
  // Input:  MxBxSxNxH (Format 0)
  // Output: MxBxNxSxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;
  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;

  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = n * H + s * NH + (b + m * batch_size) * NHS;
  const int out_offset = s * H + n * sequence_length * H + (b  + m * batch_size) * NHS;

  const int h = threadIdx.x;
  if (h < head_size) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeLarge(const int head_size, const T* input, const T* biases, T* output) {
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int stride = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;

  const int H = head_size;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;

  int in_offset = n * H + s * NH + (b + m * batch_size) * NHS;
  const int out_offset = (s + n * sequence_length) * H + (b  + m * batch_size) * NHS;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size,
    const T* input, const T* biases, T* output) {
  const dim3 grid(sequence_length, batch_size, num_matrices);
  if (head_size * num_heads <= max_threads_per_block) {
    const dim3 block(head_size, num_heads, 1);
    if (format == 1) {
      AddBiasTransposeQKV<T><<<grid, block, 0, stream>>>(input, biases, output);
    }
    else {
      AddBiasTranspose<T><<<grid, block, 0, stream>>>(input, biases, output);
    }
  } else {
    const dim3 block(CeilDiv(max_threads_per_block, num_heads), num_heads, 1);
    if (format == 1) {
      AddBiasTransposeQKVLarge<T><<<grid, block, 0, stream>>>(head_size, input, biases, output);
    }
    else{
      AddBiasTransposeLarge<T><<<grid, block, 0, stream>>>(head_size, input, biases, output);
    }
  }
}

template void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices,const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size,
    const float* input, const float* biases, float* output);

template void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size,
    const half* input, const half* biases, half* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
