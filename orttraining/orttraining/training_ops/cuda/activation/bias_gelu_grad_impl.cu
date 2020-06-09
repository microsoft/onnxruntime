// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/activation/bias_gelu_grad_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/activation/gelu_grad_impl_common.cuh"

namespace onnxruntime {
namespace cuda {

namespace {
template <typename T>
__global__ void BiasGeluGradDxKernel(
    int64_t input_size, int64_t bias_size,
    const T* dY, const T* X, const T* B, T* dX) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, input_size);
  const CUDA_LONG bias_id = id % bias_size;
  dX[id] = ComputeGeluGradScalar(dY[id], X[id] + B[bias_id]);
}

template <typename T>
__global__ void BiasGeluApproximationGradDxKernel(
    int64_t input_size, int64_t bias_size,
    const T* dY, const T* X, const T* B, T* dX) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, input_size);
  const CUDA_LONG bias_id = id % bias_size;
  dX[id] = ComputeGeluApproximationGradScalar(dY[id], X[id] + B[bias_id]);
}
}  // namespace

template <typename T>
void LaunchBiasGeluGradDxKernel(
    int64_t input_size, int64_t bias_size,
    const T* dY, const T* X, const T* B, T* dX) {
  const auto blocks_per_grid = CeilDiv(input_size, GridDim::maxThreadsPerBlock);
  BiasGeluGradDxKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock>>>(
      input_size, bias_size, dY, X, B, dX);
}

template <typename T>
void LaunchBiasGeluApproximationGradDxKernel(
    int64_t input_size, int64_t bias_size,
    const T* dY, const T* X, const T* B, T* dX) {
  const auto blocks_per_grid = CeilDiv(input_size, GridDim::maxThreadsPerBlock);
  BiasGeluApproximationGradDxKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock>>>(
      input_size, bias_size, dY, X, B, dX);
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
