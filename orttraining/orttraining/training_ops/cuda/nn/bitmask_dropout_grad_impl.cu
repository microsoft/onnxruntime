/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Modifications Copyright (c) Microsoft. */

#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/nn/bitmask_dropout_grad_impl.h"
#include <curand_kernel.h>
#include <algorithm>

constexpr int WARP_SIZE = 32;

namespace onnxruntime {
namespace cuda {

/**
 * Reference "bitmask_dropout.cc" for an explanation of how bits are packed into the mask
 */
template <typename T>
__global__ void BitmaskDropoutGradientKernel(
    const int64_t N,
    const T* dY_data,
    const uint32_t* mask_data,
    const float ratio,
    T* dX_data) {
  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;

  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG step_size = gridDim.x * blockDim.x;

  for (int64_t i = idx; i < N; i += step_size) {
    if (i < N) {
      int64_t mask_index = i / WARP_SIZE;
      int64_t bit_index = i % WARP_SIZE;
      bool mask_bit = (mask_data[mask_index] & (1 << bit_index)) != 0;
      dX_data[i] = T(float(dY_data[i]) * mask_bit * scale);
    }
  }
}

template <typename T>
void BitmaskDropoutGradientKernelImpl(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    const int64_t N,
    const T* dY_data,
    const uint32_t* mask_data,
    const float ratio,
    T* dX_data) {
  // block size should be perfectly divisble by warp size for optimized performance.
  constexpr int block_size = 256;

  const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
  const int grid_size = std::min(prop.multiProcessorCount * blocks_per_sm, static_cast<int>(CeilDiv(N, block_size)));

  BitmaskDropoutGradientKernel<T><<<grid_size, block_size, 0, stream>>>(N, dY_data, mask_data, ratio, dX_data);
}

#define SPECIALIZED_BITMASK_DROPOUT_GRAD_IMPL(T)  \
  template void BitmaskDropoutGradientKernelImpl( \
      const cudaDeviceProp& prop,                 \
      cudaStream_t stream,                        \
      const int64_t N,                            \
      const T* dY_data,                           \
      const uint32_t* mask_data,                  \
      const float ratio,                          \
      T* dX_data);

SPECIALIZED_BITMASK_DROPOUT_GRAD_IMPL(float)
SPECIALIZED_BITMASK_DROPOUT_GRAD_IMPL(double)
SPECIALIZED_BITMASK_DROPOUT_GRAD_IMPL(half)
SPECIALIZED_BITMASK_DROPOUT_GRAD_IMPL(BFloat16)

}  // namespace cuda
}  // namespace onnxruntime
