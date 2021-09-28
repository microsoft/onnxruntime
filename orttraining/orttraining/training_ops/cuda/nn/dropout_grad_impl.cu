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
#include "orttraining/training_ops/cuda/nn/dropout_grad_impl.h"
#include <curand_kernel.h>
#include <algorithm>

namespace onnxruntime {
namespace cuda {

constexpr int UNROLL = 4;

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DropoutGradientKernel(
    const int64_t N,
    const T* dY_data,
    const bool* mask_data,
    const float scale,
    T* dX_data) {

  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG id = idx * UNROLL;

  #pragma unroll
  for (int i = 0; i < UNROLL; i++) {
    CUDA_LONG li = id + i;
    if (li < N) {
      dX_data[li] = T(float(dY_data[li]) * mask_data[li] * scale);
    }
  }
}

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DropoutGradientVectorizedKernel(
    const int64_t N,
    const T* dY_data,
    const bool* mask_data,
    const float scale,
    T* dX_data) {

  // using vectorized data load/store approach when N % 4 == 0 
  // since this is typical case for input shape size
  using LoadT = aligned_vector<T, UNROLL>;
  using MaskLoadT = aligned_vector<bool, UNROLL>;

  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG id = idx * UNROLL;

  if (id < N) {
    // vectorized load into storage
    T src[UNROLL];
    LoadT *value1 = reinterpret_cast<LoadT*>(&src);
    *value1 = *reinterpret_cast<const LoadT*>(&dY_data[id]);

    bool mask[UNROLL];
    MaskLoadT *value2 = reinterpret_cast<MaskLoadT*>(&mask);
    *value2 = *reinterpret_cast<const MaskLoadT*>(&mask_data[id]);

    T r[UNROLL];

    // actual computation
    #pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      r[ii] = T(float(src[ii]) * mask[ii] * scale);
    }
    // Vectorized writes for dX_data
    *(reinterpret_cast<LoadT*>(&dX_data[id])) = *reinterpret_cast<LoadT*>(&r[0]);
  }

}

template <typename T>
void DropoutGradientKernelImpl(
    cudaStream_t stream,
    const int64_t N,
    const T* dY_data,
    const bool* mask_data,
    const float ratio,
    T* dX_data) {
  if (ratio == 0.0f) {
    if (dY_data != dX_data) {
      CUDA_CALL_THROW(cudaMemcpyAsync(dX_data, dY_data, N * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    }
  } else {
    const float scale = 1.f / (1.f - ratio);
    const int blocksPerGrid = static_cast<int>(CeilDiv(N, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
    if (N % UNROLL != 0) {
      DropoutGradientKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
                           <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(N, dY_data, mask_data, scale, dX_data);
    } else {
      DropoutGradientVectorizedKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
                                     <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(N, dY_data, mask_data, scale, dX_data);
    }

  }
}

#define SPECIALIZED_DROPOUT_GRAD_IMPL(T)   \
  template void DropoutGradientKernelImpl( \
      cudaStream_t stream,           \
      const int64_t N,                     \
      const T* dY_data,                    \
      const bool* mask_data,               \
      const float scale,                   \
      T* dX_data);

SPECIALIZED_DROPOUT_GRAD_IMPL(float)
SPECIALIZED_DROPOUT_GRAD_IMPL(double)
SPECIALIZED_DROPOUT_GRAD_IMPL(half)
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
SPECIALIZED_DROPOUT_GRAD_IMPL(nv_bfloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime
