// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cuda_fp16.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <type_traits>
#include "core/framework/float16.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
#ifdef USE_ROCM
    val += __shfl_xor(val, mask);
#else
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
#endif
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [num_tokens, hidden_size]
    const scalar_t* __restrict__ input,   // [num_tokens, hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

template <typename T>
void LaunchRMSNormKernel(
    const cudaStream_t stream,
    void* out,           // [num_tokens, hidden_size]
    const void* input,   // [num_tokens, hidden_size]
    const void* weight,  // [hidden_size]
    float epsilon,
    const int64_t* input_shape) {
  int num_tokens = input_shape[0];
  int hidden_size = input_shape[1];

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  // const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using scalar_t = T;
  rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
      (scalar_t*)out,
      (const scalar_t*)input,
      (const scalar_t*)weight,
      epsilon,
      num_tokens,
      hidden_size);
}

template
void LaunchRMSNormKernel<float>(
    const cudaStream_t stream,
    void* out,           // [num_tokens, hidden_size]
    const void* input,   // [num_tokens, hidden_size]
    const void* weight,  // [hidden_size]
    float epsilon,
    const int64_t* input_shape);

template void LaunchRMSNormKernel<half>(
    const cudaStream_t stream,
    void* out,           // [num_tokens, hidden_size]
    const void* input,   // [num_tokens, hidden_size]
    const void* weight,  // [hidden_size]
    float epsilon,
    const int64_t* input_shape);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
