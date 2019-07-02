// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "softmaxcrossentropy_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _SoftMaxCrossEntropy(
    const T* input_data,
    const T* label_data,
    T* output_data,
    CUDA_LONG N) {

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = _Log(input_data[id]) * label_data[id] * -1;
}

template <typename T>
void SoftMaxCrossEntropyImpl(
    const T* prob,
    const T* label,
    T* output_data,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _SoftMaxCrossEntropy<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      prob,
      label,
      output_data,
      N);
}

#define SPECIALIZED_IMPL_SoftMaxEntropyImpl(T) \
  template void SoftMaxCrossEntropyImpl(       \
      const T* prob,                           \
      const T* label,                          \
      T* output_data,                          \
      size_t count);

SPECIALIZED_IMPL_SoftMaxEntropyImpl(float)

template <typename T>
__global__ void _SoftMaxCrossEntropyGrad(
    const T* dY,
    const T* prob,
    const T* label,
    T* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = (prob[id] - label[id]) * (*dY);
}

template <typename T>
void SoftMaxCrossEntropyGradImpl(
    const T* dY,
    const T* prob,
    const T* label,
    T* output_data,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _SoftMaxCrossEntropyGrad<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      dY,
      prob,
      label,
      output_data,
      N);
}

#define SPECIALIZED_IMPL_SoftMaxEntropyGradImpl(T) \
  template void SoftMaxCrossEntropyGradImpl(       \
      const T* dY,                                 \
      const T* prob,                               \
      const T* label,                              \
      T* output_data,                              \
      size_t count);

SPECIALIZED_IMPL_SoftMaxEntropyGradImpl(float)

template <typename T, typename Tin>
__global__ void _SparseSoftmaxCrossEntropy(
    const T* input_data,
    const Tin* label_data,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  CUDA_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < D);
  output_data[i] = -_Log(input_data[i * D + label_data[i]]);
}

template <typename T, typename Tin>
__global__ void _WeightedSparseSoftmaxCrossEntropy(
    const T* input_data,
    const Tin* label_data,
    const T* weight_data,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  CUDA_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < D);
  output_data[i] = -_Log(input_data[i * D + label_data[i]]) * weight_data[i];
}

template <typename T, typename Tin>
void SparseSoftmaxCrossEntropyImpl(
    const T* prob,
    const Tin* label,
    const T* weight,
    T* output_data,
    size_t count,
    size_t label_depth) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG D = static_cast<CUDA_LONG>(label_depth);
  if (weight) {
    _WeightedSparseSoftmaxCrossEntropy<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      prob,
      label,
      weight,
      output_data,
      N,
      D);
  } else {
    _SparseSoftmaxCrossEntropy<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        prob,
        label,
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
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  int idx = i * D + label[i];
  output_data[idx] = prob[idx] - 1.;
  // TODO. Take dY into account
}

template <typename T, typename Tin>
__global__ void _WeightedSparseSoftmaxCrossEntropyGrad(
    const T* dY,
    const T* prob,
    const Tin* label,
    const T* weight,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N * D);
  int row = i / D;
  int d = i % D;
  output_data[i] = (*dY) * weight[row] * (prob[i] - 1.0 * (d == label[row]));
}

template <typename T, typename Tin>
void SparseSoftmaxCrossEntropyGradImpl(
    const T* dY,
    const T* prob,
    const Tin* label,
    const T* weight,
    T* output_data,
    size_t count,
    size_t label_depth) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG D = static_cast<CUDA_LONG>(label_depth);
  if (weight) {
    _WeightedSparseSoftmaxCrossEntropyGrad<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      dY,
      prob,
      label,
      weight,
      output_data,
      N,
      D);
  } else {
    _SparseSoftmaxCrossEntropyGrad<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        dY,
        prob,
        label,
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
      T* output_data,                                         \
      size_t count,                                           \
      size_t label_depth);

SPECIALIZED_IMPL_SparseSoftMaxEntropyGradImpl(float, int32_t)
SPECIALIZED_IMPL_SparseSoftMaxEntropyGradImpl(float, int64_t)

}  // namespace cuda
}  // namespace onnxruntime
