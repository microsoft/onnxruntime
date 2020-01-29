// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cu_inc/common.cuh"
#include "fft_ops_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
template <typename T>
__global__ void _Normalize(
    T* data,
    const int64_t N,
    const int64_t scale) {
  CUDA_LONG id = ::onnxruntime::cuda::GridDim::GetLinearThreadId();
  if (id >= N)
    return;

  int index = static_cast<int>(id);
  data[index] = data[index] / scale;
}

template <typename T>
void PostProcess(const FFTParams params, Tensor* Y, T* output_data) {
  int64_t scale = (std::accumulate(params.signal_dims.begin(), params.signal_dims.end(), 1ll, std::multiplies<int64_t>()));

  TensorShape output_shape = Y->Shape();
  int64_t N = output_shape.Size();
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / ::onnxruntime::cuda::GridDim::maxThreadsPerBlock));
  _Normalize<T><<<blocksPerGrid, ::onnxruntime::cuda::GridDim::maxThreadsPerBlock, 0>>>(output_data, N, scale);
}

#define SPECIALIZED_IMPL(T) \
  template void PostProcess<T>(const FFTParams, Tensor* Y, T* output_data);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
//SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
