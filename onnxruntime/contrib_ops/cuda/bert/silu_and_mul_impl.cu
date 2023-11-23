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
__device__ __forceinline__ T silu(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename scalar_t>
__global__ void silu_and_mul_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d]
    const scalar_t* __restrict__ input,  // [num_tokens, 2, d]
    const int d) {
  const int token_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = input[token_idx * 2 * d + idx];
    const scalar_t y = input[token_idx * 2 * d + d + idx];
    out[token_idx * d + idx] = silu(x) * y;
  }
}

template <typename T>
void LaunchSiluMulKernel(
    cudaStream_t stream,
    T* out,  // [num_tokens, d]
    const T* input,
    const int64_t* input_shape)  // [num_tokens, 2 * d]
{
  int num_tokens = input_shape[0];
  int d = input_shape[1] / 2;

  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  //const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = T;
  silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
      out,
      input,
      d);
}

template void LaunchSiluMulKernel<float>(cudaStream_t stream, float* out, const float* input, const int64_t* input_shape);
template void LaunchSiluMulKernel<half>(cudaStream_t stream, half* out, const half* input, const int64_t* input_shape);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
