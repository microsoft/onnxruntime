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

#define SPECIALIZED_IMPL_SoftMaxEntropyImpl(T)      \
template void SoftMaxCrossEntropyImpl(              \
    const T* prob,                                  \
    const T* label,                                 \
    T* output_data,                                 \
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
void SoftMaxCrossEntropyGradImpl(const T* dY, const T* prob, const T* label, T* output_data, size_t count) {
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

}  // namespace cuda
}  // namespace onnxruntime
