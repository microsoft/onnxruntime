// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "batch_norm_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _InterBatchNormKernel(
    const T* scale,
    const T* bias,
    const T* mean,
    const T* variance,
    const T epsilon,
    T* fused_alpha,
    T* fused_bias,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  const T alpha = scale[id] / _Sqrt(variance[id] + (T)epsilon);
  fused_alpha[id] = alpha;
  fused_bias[id] = bias[id] - alpha * mean[id];
}

template <typename T>
__global__ void _BatchNormKernel(
    const T* input_data,
    const T* fused_alpha,
    const T* fused_bias,
    const fast_divmod fdm_HWD,
    const fast_divmod fdm_C,
    T* output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int nc = fdm_HWD.div(id);
  int n, c;
  fdm_C.divmod(nc, n, c);

  output_data[id] = fused_alpha[c] * input_data[id] + fused_bias[c];
}

template <typename T>
void BatchNormImpl(
    const T* input_data,
    const T* scale,
    const T* bias,
    const T* mean,
    const T* variance,
    const T epsilon,
    const fast_divmod& fdm_HWD,
    const fast_divmod& fdm_C,
    T* fused_alpha,
    T* fused_bias,
    T* output_data,
    size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  _InterBatchNormKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      scale, bias, mean, variance, epsilon, fused_alpha, fused_bias, (CUDA_LONG)N);

  _BatchNormKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      input_data, fused_alpha, fused_bias, fdm_HWD, fdm_C, output_data, (CUDA_LONG)N);
}

#define SPECIALIZED_IMPL(T)         \
  template void BatchNormImpl<T>(   \
      const T* input_data,          \
      const T* scale,               \
      const T* bias,                \
      const T* mean,                \
      const T* variance,            \
      const T epsilon,              \
      const fast_divmod& fdm_HWD,   \
      const fast_divmod& fdm_C,     \
      T* fused_alpha,               \
      T* fused_bias,                \
      T* output_data,               \
      size_t count);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime
