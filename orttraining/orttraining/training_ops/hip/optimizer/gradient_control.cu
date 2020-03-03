// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/hip_common.h"
#include "core/providers/hip/cu_inc/common.cuh"
#include "core/providers/hip/atomic/common.cuh"
#include "gradient_control.h"

namespace onnxruntime {
namespace hip {
template <typename T, typename T_GRAD>
__global__ void _AccumulateGradient(
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  accumulated_gradient[id] = gradient_buffer[id] + T(gradient[id]);
}

template <typename T, typename T_GRAD>
void AccumulateGradientImpl(
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  HIP_LONG N = static_cast<HIP_LONG>(count);
  hipLaunchKernelGGL(_AccumulateGradient<T, T_GRAD>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
      gradient_buffer,
      gradient,
      accumulated_gradient,
      N);
}

#define SPECIALIZED_IMPL_AccumulateGradient(T, T_GRAD) \
  template void AccumulateGradientImpl(                \
      const T* gradient_buffer,                        \
      const T_GRAD* gradient,                          \
      T* accumulated_gradient,                         \
      size_t count);

SPECIALIZED_IMPL_AccumulateGradient(float, float)
SPECIALIZED_IMPL_AccumulateGradient(float, half)

}  // namespace hip
}  // namespace onnxruntime