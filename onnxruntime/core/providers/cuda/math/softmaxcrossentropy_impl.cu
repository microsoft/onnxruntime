// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "softmaxcrossentropy_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _SoftMaxCrossEntropy(
    const T* input_data,
    const T* label_data,
    CUDA_LONG NORMALIZE_FACTOR,
    T* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = -_Log(_Max(input_data[id], 1e-30f)) * label_data[id] / NORMALIZE_FACTOR;
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void SoftMaxCrossEntropyImpl(
    const T* prob,
    const T* label,
    size_t normalize_factor,
    T* output_data,
    size_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG NORMALIZE_FACTOR = static_cast<CUDA_LONG>(normalize_factor);
  _SoftMaxCrossEntropy<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
    <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
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

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _SoftMaxCrossEntropyGrad(
    const T* dY,
    const T* prob,
    const T* label,
    CUDA_LONG NORMALIZE_FACTOR,
    T* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = (prob[id] - label[id]) * (*dY) / NORMALIZE_FACTOR;
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void SoftMaxCrossEntropyGradImpl(
    const T* dY,
    const T* prob,
    const T* label,
    size_t normalize_factor,
    T* output_data,
    size_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG NORMALIZE_FACTOR = static_cast<CUDA_LONG>(normalize_factor);
  _SoftMaxCrossEntropyGrad<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
    <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
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

template <typename T, typename Tin, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _SparseSoftmaxCrossEntropy(
    const T* input_data,
    const Tin* label_data,
    const T* normalize_factor_data,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_KERNEL_ASSERT(label_data[id] >= 0 && label_data[id] < D);
      output_data[id] = -_Log(_Max(input_data[id * D + label_data[id]], 1e-30f)) / (*normalize_factor_data);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, typename Tin, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _WeightedSparseSoftmaxCrossEntropy(
    const T* input_data,
    const Tin* label_data,
    const T* weight_data,
    const T* normalize_factor_data,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_KERNEL_ASSERT(label_data[id] >= 0 && label_data[id] < D);
      output_data[id] = -_Log(_Max(input_data[id * D + label_data[id]], 1e-30f)) * weight_data[id] / (*normalize_factor_data);
      id += NumThreadsPerBlock;
    }
  }
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
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG D = static_cast<CUDA_LONG>(label_depth);
  if (weight) {
    _WeightedSparseSoftmaxCrossEntropy<T, Tin, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        prob,
        label,
        weight,
        normalize_factor,
        output_data,
        N,
        D);
  } else {
    _SparseSoftmaxCrossEntropy<T, Tin, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
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

template <typename T, typename Tin, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _SparseSoftmaxCrossEntropyGrad(
    const T* dY,
    const T* prob,
    const Tin* label,
    const T* normalize_factor,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N * D, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N * D) {
      int row = id / D;
      int d = id % D;
      output_data[id] = (*dY) * (prob[id] - 1.0 * (d == label[row])) / (*normalize_factor);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, typename Tin, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _WeightedSparseSoftmaxCrossEntropyGrad(
    const T* dY,
    const T* prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N * D, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N * D) {
      int row = id / D;
      int d = id % D;
      output_data[id] = (*dY) * weight[row] * (prob[id] - 1.0 * (d == label[row])) / (*normalize_factor);
      id += NumThreadsPerBlock;
    }
  }
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
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG D = static_cast<CUDA_LONG>(label_depth);
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  if (weight) {
    _WeightedSparseSoftmaxCrossEntropyGrad<T, Tin, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        dY,
        prob,
        label,
        weight,
        normalize_factor,
        output_data,
        N,
        D);
  } else {
    _SparseSoftmaxCrossEntropyGrad<T, Tin, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
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

}  // namespace cuda
}  // namespace onnxruntime
