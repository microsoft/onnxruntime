// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "dropout_curand_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void DropoutKernel(
  const int64_t N,
  const float ratio,
  const T* X_data,
  const float* random_data,
  T* Y_data,
  bool* mask_data) {
  const T scale = T(1.0f) / T(1.0f - ratio);
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  mask_data[id] = random_data[id] > ratio;
  Y_data[id] = X_data[id] * T(mask_data[id]) * scale;
}

template <typename T>
void DropoutKernelImpl(
  const int64_t N,
  const float ratio,
  const T* X_data,
  const float* random_data,
  T* Y_data,
  bool* mask_data) {
  int blocksPerGrid = (N + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
  DropoutKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(N, ratio, X_data, random_data, Y_data, mask_data);
}

#define DROPOUT_IMPL(T)           \
template void DropoutKernelImpl(  \
  const int64_t N,                \
  const float ratio,              \
  const T* X_data,                \
  const float* random_data,       \
  T* Y_data,                      \
  bool* mask_data);

DROPOUT_IMPL(float)
DROPOUT_IMPL(double)
DROPOUT_IMPL(half)

template <typename T>
__global__ void DropoutGradientKernel(
  const int64_t N,
  const T* dY_data,
  const bool* mask_data,
  const float scale,
  T* dX_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  dX_data[id] = dY_data[id] * T(mask_data[id] * scale);
}

template <typename T>
void DropoutGradientKernelImpl(
  const int64_t N,
  const T* dY_data,
  const bool* mask_data,
  const float scale,
  T* dX_data) {
  int blocksPerGrid = (N + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
  DropoutGradientKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(N, dY_data, mask_data, scale, dX_data);
}

#define DROPOUT_GRAD_IMPL(T)              \
template void DropoutGradientKernelImpl(  \
  const int64_t N,                        \
  const T* dY_data,                       \
  const bool* mask_data,                  \
  const float scale,                      \
  T* dX_data);

DROPOUT_GRAD_IMPL(float)
DROPOUT_GRAD_IMPL(double)
DROPOUT_GRAD_IMPL(half)
}
}
