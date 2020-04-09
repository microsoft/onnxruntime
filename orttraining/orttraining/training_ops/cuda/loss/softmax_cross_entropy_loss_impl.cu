// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/loss/softmax_cross_entropy_loss_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename Tin>
__global__ void _ComputeWeightsSoftmaxCrossEntropy(
    T* weight_data_nd,
    const Tin* label_data,
    const T* weight_data,
    CUDA_LONG N,
    CUDA_LONG D,
    CUDA_LONG ignore_index) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  CUDA_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < D);
  if (label_data[i] != ignore_index) {
    weight_data_nd[i] = weight_data != nullptr ? weight_data[label_data[i]] : 1;
  }
}

template <typename T, typename Tin>
void ComputeWeightsSoftmaxCrossEntropyImpl(
    T* weight_data_nd,
    const Tin* label,
    const T* weight,
    size_t count,
    size_t label_depth,
    int64_t ignore_index) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG D = static_cast<CUDA_LONG>(label_depth);
  CUDA_LONG II = static_cast<CUDA_LONG>(ignore_index);
  _ComputeWeightsSoftmaxCrossEntropy<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      weight_data_nd,
      label,
      weight,
      count,
      D,
      II);
}

template <typename T, typename Tin>
__global__ void _WeightedSoftmaxCrossEntropyLoss(
    const T* log_prob_data,
    const Tin* label_data,
    const T* weight_data,
    const T* normalize_factor_data,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D,
    CUDA_LONG II) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  if (II == label_data[i]) {
    output_data[i] = 0;
  } else {
    CUDA_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < D);
    output_data[i] = -log_prob_data[i * D + label_data[i]] * weight_data[i] / (*normalize_factor_data);
  }
}

template <typename T, typename Tin>
void SoftmaxCrossEntropyLossImpl(
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    size_t count,
    size_t label_depth,
    int64_t ignore_index) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG D = static_cast<CUDA_LONG>(label_depth);
  CUDA_LONG II = static_cast<CUDA_LONG>(ignore_index);
  _WeightedSoftmaxCrossEntropyLoss<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      log_prob,
      label,
      weight,
      normalize_factor,
      output_data,
      N,
      D,
      II);
}

#define SPECIALIZED_IMPL_SoftMaxEntropyLossImpl(T, Tin) \
  template void SoftmaxCrossEntropyLossImpl(            \
      const T* log_prob,                                \
      const Tin* label,                                 \
      const T* weight,                                  \
      const T* normalize_factor,                        \
      T* output_data,                                   \
      size_t count,                                     \
      size_t label_depth,                               \
      int64_t ignore_index);

SPECIALIZED_IMPL_SoftMaxEntropyLossImpl(float, int32_t)
SPECIALIZED_IMPL_SoftMaxEntropyLossImpl(float, int64_t)

template <typename T, typename Tin>
__global__ void _WeightedSoftmaxCrossEntropyLossGrad(
    const T* dY,
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D,
    CUDA_LONG II) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N * D);

  int row = i / D;
  int d = i % D;
  if (II == label[row]) {
    output_data[i] = 0;
  } else {
    output_data[i] = (*dY) * weight[row] * (_Exp(log_prob[i]) - 1.0 * (d == label[row])) / (*normalize_factor);
  }
}

template <typename T, typename Tin>
__global__ void _WeightedReductionNoneSoftmaxCrossEntropyLossGrad(
    const T* dY,
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D,
    CUDA_LONG II) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N * D);

  int row = i / D;
  int d = i % D;
  if (II == label[row]) {
    output_data[i] = 0;
  } else {
    output_data[i] = dY[row] * weight[row] * (_Exp(log_prob[i]) - 1.0 * (d == label[row])) / (*normalize_factor);
  }
}

template <typename T, typename Tin>
void SoftmaxCrossEntropyLossGradImpl(
    const T* dY,
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    size_t count,
    size_t label_depth,
    int64_t ignore_index,
    bool reduction_none) {
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG D = static_cast<CUDA_LONG>(label_depth);
  CUDA_LONG II = static_cast<CUDA_LONG>(ignore_index);
  int blocksPerGrid = (int)(ceil(static_cast<float>(N * D) / GridDim::maxThreadsPerBlock));
  if (reduction_none) {
    _WeightedReductionNoneSoftmaxCrossEntropyLossGrad<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        dY,
        log_prob,
        label,
        weight,
        normalize_factor,
        output_data,
        N,
        D,
        II);
  } else {
    _WeightedSoftmaxCrossEntropyLossGrad<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        dY,
        log_prob,
        label,
        weight,
        normalize_factor,
        output_data,
        N,
        D,
        II);
  }
}

#define SPECIALIZED_IMPL_SoftMaxEntropyLossGradImpl(T, Tin) \
  template void SoftmaxCrossEntropyLossGradImpl(            \
      const T* dY,                                          \
      const T* log_prob,                                    \
      const Tin* label,                                     \
      const T* weight,                                      \
      const T* normalize_factor,                            \
      T* output_data,                                       \
      size_t count,                                         \
      size_t label_depth,                                   \
      int64_t ignore_index,                                 \
      bool reducation_none);

SPECIALIZED_IMPL_SoftMaxEntropyLossGradImpl(float, int32_t)
SPECIALIZED_IMPL_SoftMaxEntropyLossGradImpl(float, int64_t)

#define SPECIALIZED_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(T, Tin) \
  template void ComputeWeightsSoftmaxCrossEntropyImpl(                 \
      T* weight_data_nd,                                               \
      const Tin* label,                                                \
      const T* weight,                                                 \
      size_t count,                                                    \
      size_t label_depth,                                              \
      int64_t ignore_index);

SPECIALIZED_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(float, int32_t)
SPECIALIZED_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(float, int64_t)

}  // namespace cuda
}  // namespace onnxruntime