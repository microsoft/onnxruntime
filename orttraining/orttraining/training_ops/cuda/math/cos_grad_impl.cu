// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cos_grad_impl.h"
#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _CosGradImpl(const T* dy, const T* Y, T* output, CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output[id] = -1 * dy[id] * sin(Y[id]);
}

template <typename T>
void CosGradImpl(cudaStream_t stream, const T* dy, const T* Y, T* output, size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _CosGradImpl<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(dy, Y, output, N);
}

#define SPECIALIZE_COSGRAD_IMPL(T) \
  template void CosGradImpl(cudaStream_t stream, const T* dy, const T* Y, T* output, size_t count);

SPECIALIZE_COSGRAD_IMPL(half)
SPECIALIZE_COSGRAD_IMPL(float)
SPECIALIZE_COSGRAD_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime
