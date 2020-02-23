#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/cu_inc/common.cuh"
#include "softmaxcrossentropy_impl.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

template <typename T>
__global__ void _SoftMaxCrossEntropy(
    const T* input_data,
    const T* label_data,
    HIP_LONG NORMALIZE_FACTOR,
    T* output_data,
    HIP_LONG N) {

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = -_Log(_Max(input_data[id], 1e-30f)) * label_data[id] / NORMALIZE_FACTOR;
}

template <typename T>
void SoftMaxCrossEntropyImpl(
    const T* prob,
    const T* label,
    size_t normalize_factor,
    T* output_data,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  HIP_LONG N = static_cast<HIP_LONG>(count);
  HIP_LONG NORMALIZE_FACTOR = static_cast<HIP_LONG>(normalize_factor);
  hipLaunchKernelGGL(_SoftMaxCrossEntropy<T>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
      prob,
      label,
      NORMALIZE_FACTOR,
      output_data,
      N);
}

#define SPECIALIZED_IMPL_SoftMaxEntropyImpl(T) \
  template void SoftMaxCrossEntropyImpl(       \
      const T* prob,                           \
      const T* label,                          \
      size_t normalize_factor,                 \
      T* output_data,                          \
      size_t count);

SPECIALIZED_IMPL_SoftMaxEntropyImpl(float)

template <typename T>
__global__ void _SoftMaxCrossEntropyGrad(
    const T* dY,
    const T* prob,
    const T* label,
    HIP_LONG NORMALIZE_FACTOR,
    T* output_data,
    HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = (prob[id] - label[id]) * (*dY) / NORMALIZE_FACTOR;
}

template <typename T>
void SoftMaxCrossEntropyGradImpl(
    const T* dY,
    const T* prob,
    const T* label,
    size_t normalize_factor,
    T* output_data,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  HIP_LONG N = static_cast<HIP_LONG>(count);
  HIP_LONG NORMALIZE_FACTOR = static_cast<HIP_LONG>(normalize_factor);
  hipLaunchKernelGGL(_SoftMaxCrossEntropyGrad<T>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
      dY,
      prob,
      label,
      NORMALIZE_FACTOR,
      output_data,
      N);
}

#define SPECIALIZED_IMPL_SoftMaxEntropyGradImpl(T) \
  template void SoftMaxCrossEntropyGradImpl(       \
      const T* dY,                                 \
      const T* prob,                               \
      const T* label,                              \
      size_t normalize_factor,                     \
      T* output_data,                              \
      size_t count);

SPECIALIZED_IMPL_SoftMaxEntropyGradImpl(float)

template <typename T, typename Tin>
__global__ void _SparseSoftmaxCrossEntropy(
    const T* input_data,
    const Tin* label_data,
    const T* normalize_factor_data,
    T* output_data,
    HIP_LONG N,
    HIP_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  HIP_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < D);
  output_data[i] = -_Log(_Max(input_data[i * D + label_data[i]], 1e-30f)) / (*normalize_factor_data);
}

template <typename T, typename Tin>
__global__ void _WeightedSparseSoftmaxCrossEntropy(
    const T* input_data,
    const Tin* label_data,
    const T* weight_data,
    const T* normalize_factor_data,
    T* output_data,
    HIP_LONG N,
    HIP_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  HIP_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < D);
  output_data[i] = -_Log(_Max(input_data[i * D + label_data[i]], 1e-30f)) * weight_data[i] / (*normalize_factor_data);
}

template <typename T, typename Tin>
void SparseSoftmaxCrossEntropyImpl(
    const T* prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    size_t count,
    size_t label_depth) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  HIP_LONG N = static_cast<HIP_LONG>(count);
  HIP_LONG D = static_cast<HIP_LONG>(label_depth);
  if (weight) {
    hipLaunchKernelGGL(_WeightedSparseSoftmaxCrossEntropy<T, Tin>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
      prob,
      label,
      weight,
      normalize_factor,
      output_data,
      N,
      D);
  } else {
    hipLaunchKernelGGL(_SparseSoftmaxCrossEntropy<T, Tin>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
        prob,
        label,
        normalize_factor,
        output_data,
        N,
        D);
  }
}

#define SPECIALIZED_IMPL_SparseSoftMaxEntropyImpl(T, Tin) \
  template void SparseSoftmaxCrossEntropyImpl(            \
      const T* prob,                                      \
      const Tin* label,                                   \
      const T* weight,                                    \
      const T* normalize_factor,                          \
      T* output_data,                                     \
      size_t count,                                       \
      size_t label_depth);

SPECIALIZED_IMPL_SparseSoftMaxEntropyImpl(float, int32_t)
SPECIALIZED_IMPL_SparseSoftMaxEntropyImpl(float, int64_t)

template <typename T, typename Tin>
__global__ void _SparseSoftmaxCrossEntropyGrad(
    const T* dY,
    const T* prob,
    const Tin* label,
    const T* normalize_factor,
    T* output_data,
    HIP_LONG N,
    HIP_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N * D);
  int row = i / D;
  int d = i % D;
  output_data[i] = (*dY) * (prob[i] - 1.0 * (d == label[row])) / (*normalize_factor);
}

template <typename T, typename Tin>
__global__ void _WeightedSparseSoftmaxCrossEntropyGrad(
    const T* dY,
    const T* prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    HIP_LONG N,
    HIP_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N * D);
  int row = i / D;
  int d = i % D;
  output_data[i] = (*dY) * weight[row] * (prob[i] - 1.0 * (d == label[row])) / (*normalize_factor);
}

template <typename T, typename Tin>
void SparseSoftmaxCrossEntropyGradImpl(
    const T* dY,
    const T* prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    size_t count,
    size_t label_depth) {
  HIP_LONG N = static_cast<HIP_LONG>(count);
  HIP_LONG D = static_cast<HIP_LONG>(label_depth);
  int blocksPerGrid = (int)(ceil(static_cast<float>(N * D) / GridDim::maxThreadsPerBlock));
  if (weight) {
    hipLaunchKernelGGL(_WeightedSparseSoftmaxCrossEntropyGrad<T, Tin>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
      dY,
      prob,
      label,
      weight,
      normalize_factor,
      output_data,
      N,
      D);
  } else {
    hipLaunchKernelGGL(_SparseSoftmaxCrossEntropyGrad<T, Tin>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
        dY,
        prob,
        label,
        normalize_factor,
        output_data,
        N,
        D);
  }
}

#define SPECIALIZED_IMPL_SparseSoftMaxEntropyGradImpl(T, Tin) \
  template void SparseSoftmaxCrossEntropyGradImpl(            \
      const T* dY,                                            \
      const T* prob,                                          \
      const Tin* label,                                       \
      const T* weight,                                        \
      const T* normalize_factor,                              \
      T* output_data,                                         \
      size_t count,                                           \
      size_t label_depth);

SPECIALIZED_IMPL_SparseSoftMaxEntropyGradImpl(float, int32_t)
SPECIALIZED_IMPL_SparseSoftMaxEntropyGradImpl(float, int64_t)

}  // namespace hip
}  // namespace onnxruntime
