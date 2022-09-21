// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "instance_norm_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T1, typename T2>
__global__ void _InstanceNormKernel(
    const T1* __restrict__ input_data,
    const T1* __restrict__ scale,
    const T1* __restrict__ bias,
    const T2* __restrict__ mean,
    const T2* __restrict__ variance,
    const double variance_correction,
    const double epsilon,
    const fast_divmod fdm_HW,
    const fast_divmod fdm_C,
    T1* __restrict__ output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int nc = fdm_HW.div(id);
  int n, c;
  fdm_C.divmod(nc, n, c);

  // Y = scale * (x - mean) / sqrt (std * std + epsilon) + B
  output_data[id] = scale[c] * (input_data[id] - (T1)mean[nc]) / _Sqrt((T1)variance[nc] * (T1)variance_correction + (T1)epsilon) + bias[c];
}

template <typename T1, typename T2>
void InstanceNormImpl(
    cudaStream_t stream,
    const T1* input_data,
    const T1* scale,
    const T1* bias,
    const T2* mean,
    const T2* variance,
    const double variance_correction,
    const double epsilon,
    const fast_divmod& fdm_HW,
    const fast_divmod& fdm_C,
    T1* output_data,
    size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _InstanceNormKernel<T1, T2><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data, scale, bias, mean, variance, variance_correction, epsilon, fdm_HW, fdm_C, output_data, (CUDA_LONG)N);
}

#define SPECIALIZED_IMPL(T1, T2) \
  template void InstanceNormImpl<T1, T2>(cudaStream_t stream, const T1* input_data, const T1* scale, const T1* bias, const T2* mean, const T2* stddev, const double variance_correction, const double epsilon, const fast_divmod& fdm_HW, const fast_divmod& fdm_C, T1* output_data, size_t count);

SPECIALIZED_IMPL(float, float)
SPECIALIZED_IMPL(double, double)
// When the input data type is float16, the means and variances will flow in as float32 (special case)
SPECIALIZED_IMPL(half, float)

}  // namespace cuda
}  // namespace onnxruntime
