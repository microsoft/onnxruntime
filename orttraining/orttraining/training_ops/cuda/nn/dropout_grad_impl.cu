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

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DropoutGradientKernel(
    const int64_t N,
    const T* dY_data,
    const bool* mask_data,
    const float scale,
    T* dX_data) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      dX_data[id] = T(float(dY_data[id]) * mask_data[id] * scale);
      id += NumThreadsPerBlock;
    }
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
    DropoutGradientKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
                         <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(N, dY_data, mask_data, scale, dX_data);
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
