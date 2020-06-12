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
__global__ void BiasGeluGradDxKernel(
    CUDA_LONG input_size, fast_divmod bias_size_fdm,
    const T* dY, const T* X, const T* B, T* dX) {
  const auto& num_threads_per_block = blockDim.x;
  const int& bias_size = bias_size_fdm.d_;
  CUDA_LONG base_idx =
      num_consecutive_elements_per_group * num_groups_per_thread * num_threads_per_block * blockIdx.x +
      num_consecutive_elements_per_group * threadIdx.x;

#pragma unroll
  for (int i = 0; i < num_groups_per_thread; ++i) {
    if (base_idx < input_size) {
      int q, r;
      bias_size_fdm.divmod(base_idx, q, r);
      const int& base_bias_idx = r;

#pragma unroll
      for (int element_idx = 0; element_idx < num_consecutive_elements_per_group; ++element_idx) {
        const int input_idx = base_idx + element_idx;
        if (base_idx < input_size) {
          const int bias_idx =
              base_bias_idx + element_idx - static_cast<int>((base_bias_idx + element_idx) >= bias_size) * bias_size;
          dX[input_idx] = ComputeGeluGradScalar(dY[input_idx], X[input_idx] + B[bias_idx], GeluComputationMode{});
        }
      }

      base_idx += num_consecutive_elements_per_group * num_threads_per_block;
    }
  }
}

template <typename T, typename GeluComputationMode>
void LaunchBiasGeluGradDxKernel(
    int64_t input_size, int64_t bias_size,
    const T* dY, const T* X, const T* B, T* dX) {
  constexpr int num_consecutive_elements_per_group = 4;
  constexpr int num_groups_per_thread = 4;
  constexpr int num_threads_per_block = GridDim::maxThreadsPerBlock;
  const auto num_blocks_per_grid = CeilDiv(
      input_size,
      num_threads_per_block * num_consecutive_elements_per_group * num_groups_per_thread);
  const fast_divmod bias_size_fdm{static_cast<int>(bias_size)};

  BiasGeluGradDxKernel<T, GeluComputationMode, num_consecutive_elements_per_group, num_groups_per_thread>
      <<<num_blocks_per_grid, num_threads_per_block>>>(
          static_cast<CUDA_LONG>(input_size), bias_size_fdm, dY, X, B, dX);
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
