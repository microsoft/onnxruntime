// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/activation/bias_gelu_grad_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "orttraining/training_ops/cuda/activation/gelu_grad_impl_common.cuh"

namespace onnxruntime {
namespace cuda {

namespace {
template <bool use_approximation>
struct GeluGradDxScalarComputer {
  template <typename T>
  __device__ T operator()(const T dY, const T X, const T B) {
    if (use_approximation) {
      return ComputeGeluApproximationGradScalar(dY, X + B);
    } else {
      return ComputeGeluGradScalar(dY, X + B);
    }
  }
};

template <int NumElementsPerThread, typename T, typename ComputeGeluGradDxScalarFn>
__global__ void BiasGeluGradDxKernel(
    ComputeGeluGradDxScalarFn compute_gelu_grad_dx_scalar_fn,
    CUDA_LONG input_size, fast_divmod bias_size_fdm,
    const T* dY, const T* X, const T* B, T* dX) {
  const auto num_threads_per_block = blockDim.x;
  const CUDA_LONG start = NumElementsPerThread * num_threads_per_block * blockIdx.x + threadIdx.x;

  CUDA_LONG id = start;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; ++i) {
    if (id < input_size) {
      int q, r;
      bias_size_fdm.divmod(input_size, q, r);
      const int& bias_id = r;

      dX[id] = compute_gelu_grad_dx_scalar_fn(dY[id], X[id], B[bias_id]);

      id += num_threads_per_block;
    }
  }
}
}  // namespace

template <typename T>
void LaunchBiasGeluGradDxKernel(
    int64_t input_size, int64_t bias_size,
    const T* dY, const T* X, const T* B, T* dX) {
  constexpr int num_elements_per_thread = GridDim::maxThreadsPerBlock;
  const auto num_blocks_per_grid = CeilDiv(
      input_size, GridDim::maxThreadsPerBlock * num_elements_per_thread);
  const fast_divmod bias_size_fdm{static_cast<int>(bias_size)};
  BiasGeluGradDxKernel<num_elements_per_thread><<<num_blocks_per_grid, GridDim::maxThreadsPerBlock>>>(
      GeluGradDxScalarComputer<false>{},
      static_cast<CUDA_LONG>(input_size), bias_size_fdm, dY, X, B, dX);
}

template <typename T>
void LaunchBiasGeluApproximationGradDxKernel(
    int64_t input_size, int64_t bias_size,
    const T* dY, const T* X, const T* B, T* dX) {
  constexpr int num_elements_per_thread = GridDim::maxThreadsPerBlock;
  const auto num_blocks_per_grid = CeilDiv(
      input_size, GridDim::maxThreadsPerBlock * num_elements_per_thread);
  const fast_divmod bias_size_fdm{static_cast<int>(bias_size)};
  BiasGeluGradDxKernel<num_elements_per_thread><<<num_blocks_per_grid, GridDim::maxThreadsPerBlock>>>(
      GeluGradDxScalarComputer<true>{},
      static_cast<CUDA_LONG>(input_size), bias_size_fdm, dY, X, B, dX);
}

// explicit instantiations
#define SPECIALIZED_BIAS_GELU_GRAD_IMPL(T)   \
  template void LaunchBiasGeluGradDxKernel(  \
      int64_t input_size, int64_t bias_size, \
      const T* dY, const T* X, const T* B, T* dX)

SPECIALIZED_BIAS_GELU_GRAD_IMPL(half);
SPECIALIZED_BIAS_GELU_GRAD_IMPL(float);
SPECIALIZED_BIAS_GELU_GRAD_IMPL(double);

#undef SPECIALIZED_BIAS_GELU_GRAD_IMPL

#define SPECIALIZED_BIAS_GELU_APPROXIMATION_GRAD_IMPL(T) \
  template void LaunchBiasGeluApproximationGradDxKernel( \
      int64_t input_size, int64_t bias_size,             \
      const T* dY, const T* X, const T* B, T* dX)

SPECIALIZED_BIAS_GELU_APPROXIMATION_GRAD_IMPL(half);
SPECIALIZED_BIAS_GELU_APPROXIMATION_GRAD_IMPL(float);
SPECIALIZED_BIAS_GELU_APPROXIMATION_GRAD_IMPL(double);

#undef SPECIALIZED_BIAS_GELU_APPROXIMATION_GRAD_IMPL

}  // namespace cuda
}  // namespace onnxruntime
