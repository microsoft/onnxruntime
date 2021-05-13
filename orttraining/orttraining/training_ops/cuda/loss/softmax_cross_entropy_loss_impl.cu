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
    CUDA_LONG N_D,
    CUDA_LONG C,
    CUDA_LONG ignore_index) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N_D);
  const T ONE_T = 1;
  if (label_data[i] != ignore_index) {
    CUDA_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < C);
    weight_data_nd[i] = weight_data != nullptr ? weight_data[label_data[i]] : ONE_T;
  }
}

template <typename T, typename Tin>
void ComputeWeightsSoftmaxCrossEntropyImpl(
    cudaStream_t stream,
    const Tin* label,
    const T* weight,
    size_t count,
    size_t label_depth,
    int64_t ignore_index,
    T* weight_data_nd) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N_D = static_cast<CUDA_LONG>(count);
  CUDA_LONG C = static_cast<CUDA_LONG>(label_depth);
  CUDA_LONG II = static_cast<CUDA_LONG>(ignore_index);
  _ComputeWeightsSoftmaxCrossEntropy<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      weight_data_nd,
      label,
      weight,
      N_D,
      C,
      II);
}

template <typename T, typename TAcc, typename Tin>
__global__ void _WeightedSoftmaxCrossEntropyLoss(
    const T* log_prob_data,
    const Tin* label_data,
    const T* weight_data,
    const TAcc* normalize_factor_data,
    T* output_data,
    CUDA_LONG N_D,
    CUDA_LONG C,
    CUDA_LONG II) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N_D);
  if (II == label_data[i]) {
    output_data[i] = 0;
  } else {
    CUDA_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < C);
    output_data[i] = static_cast<T>(static_cast<TAcc>(-log_prob_data[i * C + label_data[i]] * weight_data[i]) /
                                    *normalize_factor_data);
  }
}

template <typename T, typename TAcc, typename Tin>
void SoftmaxCrossEntropyLossImpl(
    cudaStream_t stream,
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const TAcc* normalize_factor,
    size_t count,
    size_t label_depth,
    int64_t ignore_index,
    T* output_data) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N_D = static_cast<CUDA_LONG>(count);
  CUDA_LONG C = static_cast<CUDA_LONG>(label_depth);
  CUDA_LONG II = static_cast<CUDA_LONG>(ignore_index);
  _WeightedSoftmaxCrossEntropyLoss<T, TAcc, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      log_prob,
      label,
      weight,
      normalize_factor,
      output_data,
      N_D,
      C,
      II);
}

#define INSTANTIATE_IMPL_SoftMaxEntropyLossImpl(T, TAcc, Tin) \
  template void SoftmaxCrossEntropyLossImpl(                  \
      cudaStream_t stream,                                    \
      const T* log_prob,                                      \
      const Tin* label,                                       \
      const T* weight,                                        \
      const TAcc* normalize_factor,                           \
      size_t count,                                           \
      size_t label_depth,                                     \
      int64_t ignore_index,                                   \
      T* output_data);

INSTANTIATE_IMPL_SoftMaxEntropyLossImpl(float, float, int32_t)
INSTANTIATE_IMPL_SoftMaxEntropyLossImpl(float, float, int64_t)
INSTANTIATE_IMPL_SoftMaxEntropyLossImpl(half, float, int64_t)

template <typename T, typename TAcc, typename Tin>
__global__ void _WeightedSoftmaxCrossEntropyLossGrad(
    const T* dY,
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const TAcc* normalize_factor,
    T* output_data,
    CUDA_LONG N_D,
    CUDA_LONG C) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N_D * C);

  int row = i / C;
  int d = i % C;
  const T ZERO_T = 0;
  const TAcc ZERO_TAcc = 0;
  const TAcc ONE_TAcc = 1;
  CUDA_KERNEL_ASSERT(weight[row] == ZERO_T || (label[row] >= 0 && label[row] < C));
  if (ZERO_TAcc == *normalize_factor) {
    // normalize_factor is sum of labels' weights. Because zero
    // sum implies all weights are 0, the loss function should
    // be constant 0 and its corresponding gradient should be 0 as well.
    output_data[i] = ZERO_T;
  } else {
    output_data[i] = static_cast<T>(static_cast<TAcc>((*dY) * weight[row]) *
                                    (_Exp(static_cast<TAcc>(log_prob[i])) - ONE_TAcc * (TAcc)(d == label[row])) /
                                    (*normalize_factor));
  }
}

template <typename T, typename TAcc, typename Tin>
__global__ void _WeightedReductionNoneSoftmaxCrossEntropyLossGrad(
    const T* dY,
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const TAcc* normalize_factor,
    T* output_data,
    CUDA_LONG N_D,
    CUDA_LONG C) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N_D * C);

  int row = i / C;
  int d = i % C;
  const T ZERO_T = 0;
  const TAcc ZERO_TAcc = 0;
  const TAcc ONE_TAcc = 1;
  CUDA_KERNEL_ASSERT(weight[row] == ZERO_T || (label[row] >= 0 && label[row] < C));
  if (ZERO_TAcc == *normalize_factor) {
    // normalize_factor is sum of labels' weights. Because zero
    // sum implies all weights are 0, the loss function should
    // be constant 0 and its corresponding gradient should be 0 as well.
    output_data[i] = ZERO_T;
  } else {
    output_data[i] = static_cast<T>(static_cast<TAcc>(dY[row] * weight[row]) *
                                    (_Exp(static_cast<TAcc>(log_prob[i])) - ONE_TAcc * (TAcc)(d == label[row])) /
                                    (*normalize_factor));
  }
}

template <typename T, typename TAcc, typename Tin>
void SoftmaxCrossEntropyLossGradImpl(
    cudaStream_t stream,
    const T* dY,
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const TAcc* normalize_factor,
    size_t count,
    size_t label_depth,
    bool reduction_none,
    T* output_data) {
  CUDA_LONG N_D = static_cast<CUDA_LONG>(count);
  CUDA_LONG C = static_cast<CUDA_LONG>(label_depth);
  int blocksPerGrid = (int)(ceil(static_cast<float>(N_D * C) / GridDim::maxThreadsPerBlock));
  if (reduction_none) {
    _WeightedReductionNoneSoftmaxCrossEntropyLossGrad<T, TAcc, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        dY,
        log_prob,
        label,
        weight,
        normalize_factor,
        output_data,
        N_D,
        C);
  } else {
    _WeightedSoftmaxCrossEntropyLossGrad<T, TAcc, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        dY,
        log_prob,
        label,
        weight,
        normalize_factor,
        output_data,
        N_D,
        C);
  }
}

#define INSTANTIATE_IMPL_SoftMaxEntropyLossGradImpl(T, TAcc, Tin) \
  template void SoftmaxCrossEntropyLossGradImpl(                  \
      cudaStream_t stream,                                        \
      const T* dY,                                                \
      const T* log_prob,                                          \
      const Tin* label,                                           \
      const T* weight,                                            \
      const TAcc* normalize_factor,                               \
      size_t count,                                               \
      size_t label_depth,                                         \
      bool reducation_none,                                       \
      T* output_data);

INSTANTIATE_IMPL_SoftMaxEntropyLossGradImpl(float, float, int32_t)
INSTANTIATE_IMPL_SoftMaxEntropyLossGradImpl(float, float, int64_t)
INSTANTIATE_IMPL_SoftMaxEntropyLossGradImpl(half, float, int64_t)

#define INSTANTIATE_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(T, Tin) \
  template void ComputeWeightsSoftmaxCrossEntropyImpl(                 \
      cudaStream_t stream,                                             \
      const Tin* label,                                                \
      const T* weight,                                                 \
      size_t count,                                                    \
      size_t label_depth,                                              \
      int64_t ignore_index,                                            \
      T* weight_data_nd);

INSTANTIATE_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(float, int32_t)
INSTANTIATE_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(float, int64_t)
INSTANTIATE_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(half, int64_t)

}  // namespace cuda
}  // namespace onnxruntime