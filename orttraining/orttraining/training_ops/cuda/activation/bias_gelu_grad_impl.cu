// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/activation/bias_gelu_grad_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "orttraining/training_ops/cpu/activation/gelu_computation_mode.h"
#include "orttraining/training_ops/cuda/activation/gelu_grad_impl_common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T, typename GeluComputationMode, int num_consecutive_elements_per_group, int num_groups_per_thread>
__global__ void BiasGeluGradDxKernel(int64_t bias_size, const T* dY, const T* X, const T* B, T* dX) {
  const int64_t input_base_idx = bias_size * blockIdx.x + num_consecutive_elements_per_group * threadIdx.x;
  const int64_t bias_base_idx = num_consecutive_elements_per_group * threadIdx.x;
  const int64_t group_stride = num_consecutive_elements_per_group * blockDim.x;

#pragma unroll
  for (int group_idx = 0; group_idx < num_groups_per_thread; ++group_idx) {
#pragma unroll
    for (int element_idx = 0; element_idx < num_consecutive_elements_per_group; ++element_idx) {
      const auto offset = group_stride * group_idx + element_idx;
      if (bias_base_idx + offset < bias_size) {
        dX[input_base_idx + offset] = ComputeGeluGradScalar(
            dY[input_base_idx + offset], X[input_base_idx + offset] + B[bias_base_idx + offset],
            GeluComputationMode{});
      }
    }
  }
}

template <typename T, typename GeluComputationMode>
void LaunchBiasGeluGradDxKernel(
    int64_t input_size, int64_t bias_size,
    const T* dY, const T* X, const T* B, T* dX) {
  // each block handles bias_size elements
  // there are input_size / bias_size blocks
  constexpr int num_consecutive_elements_per_group = 4;
  constexpr int num_groups_per_thread = 4;

  const auto num_threads_per_block = CeilDiv(bias_size, num_consecutive_elements_per_group * num_groups_per_thread);
  const auto num_blocks_per_grid = input_size / bias_size;

  BiasGeluGradDxKernel<T, GeluComputationMode, num_consecutive_elements_per_group, num_groups_per_thread>
      <<<num_blocks_per_grid, num_threads_per_block>>>(bias_size, dY, X, B, dX);
}

// explicit instantiations
#define SPECIALIZED_BIAS_GELU_GRAD_IMPL(T, GeluComputationMode)     \
  template void LaunchBiasGeluGradDxKernel<T, GeluComputationMode>( \
      int64_t input_size, int64_t bias_size,                        \
      const T* dY, const T* X, const T* B, T* dX)

SPECIALIZED_BIAS_GELU_GRAD_IMPL(half, gelu_computation_mode::Default);
SPECIALIZED_BIAS_GELU_GRAD_IMPL(float, gelu_computation_mode::Default);
SPECIALIZED_BIAS_GELU_GRAD_IMPL(double, gelu_computation_mode::Default);

SPECIALIZED_BIAS_GELU_GRAD_IMPL(half, gelu_computation_mode::Approximation);
SPECIALIZED_BIAS_GELU_GRAD_IMPL(float, gelu_computation_mode::Approximation);
SPECIALIZED_BIAS_GELU_GRAD_IMPL(double, gelu_computation_mode::Approximation);

#undef SPECIALIZED_BIAS_GELU_GRAD_IMPL

}  // namespace cuda
}  // namespace onnxruntime
