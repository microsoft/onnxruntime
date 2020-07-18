#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/cu_inc/common.cuh"
#include "orttraining/training_ops/hip/loss/softmax_cross_entropy_loss_impl.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

template <typename T, typename Tin>
__global__ void _ComputeWeightsSoftmaxCrossEntropy(
    T* weight_data_nd,
    const Tin* label_data,
    const T* weight_data,
    HIP_LONG N_D,
    HIP_LONG C,
    HIP_LONG ignore_index) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N_D);
  if (label_data[i] != ignore_index) {
    HIP_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < C);
    weight_data_nd[i] = weight_data != nullptr ? weight_data[label_data[i]] : 1;
  }
}

template <typename T, typename Tin>
void ComputeWeightsSoftmaxCrossEntropyImpl(
    const Tin* label,
    const T* weight,
    size_t count,
    size_t label_depth,
    int64_t ignore_index,
    T* weight_data_nd) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  HIP_LONG N_D = static_cast<HIP_LONG>(count);
  HIP_LONG C = static_cast<HIP_LONG>(label_depth);
  HIP_LONG II = static_cast<HIP_LONG>(ignore_index);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(_ComputeWeightsSoftmaxCrossEntropy<T, Tin>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
      weight_data_nd,
      label,
      weight,
      N_D,
      C,
      II);
}

template <typename T, typename Tin>
__global__ void _WeightedSoftmaxCrossEntropyLoss(
    const T* log_prob_data,
    const Tin* label_data,
    const T* weight_data,
    const T* normalize_factor_data,
    T* output_data,
    HIP_LONG N_D,
    HIP_LONG C,
    HIP_LONG II) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N_D);
  if (II == label_data[i]) {
    output_data[i] = 0;
  } else {
    HIP_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < C);
    output_data[i] = -log_prob_data[i * C + label_data[i]] * weight_data[i] / (*normalize_factor_data);
  }
}

template <typename T, typename Tin>
void SoftmaxCrossEntropyLossImpl(
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    size_t count,
    size_t label_depth,
    int64_t ignore_index,
    T* output_data) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  HIP_LONG N_D = static_cast<HIP_LONG>(count);
  HIP_LONG C = static_cast<HIP_LONG>(label_depth);
  HIP_LONG II = static_cast<HIP_LONG>(ignore_index);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(_WeightedSoftmaxCrossEntropyLoss<T, Tin>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
      log_prob,
      label,
      weight,
      normalize_factor,
      output_data,
      N_D,
      C,
      II);
}

#define SPECIALIZED_IMPL_SoftMaxEntropyLossImpl(T, Tin) \
  template void SoftmaxCrossEntropyLossImpl(            \
      const T* log_prob,                                \
      const Tin* label,                                 \
      const T* weight,                                  \
      const T* normalize_factor,                        \
      size_t count,                                     \
      size_t label_depth,                               \
      int64_t ignore_index,                             \
      T* output_data);

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
    HIP_LONG N_D,
    HIP_LONG C) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N_D * C);

  int row = i / C;
  int d = i % C;
  HIP_KERNEL_ASSERT(weight[row] == 0 || (label[row] >= 0 && label[row] < C));
  if(0 == *normalize_factor){
    // normalize_factor is sum of labels' weights. Because zero 
    // sum implies all weights are 0, the loss function should 
    // be constant 0 and its corresponding gradient should be 0 as well.
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
    HIP_LONG N_D,
    HIP_LONG C) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N_D * C);

  int row = i / C;
  int d = i % C;
  HIP_KERNEL_ASSERT(weight[row] == 0 || (label[row] >= 0 && label[row] < C));
  if(0 == *normalize_factor){
    // normalize_factor is sum of labels' weights. Because zero 
    // sum implies all weights are 0, the loss function should 
    // be constant 0 and its corresponding gradient should be 0 as well.
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
    size_t count,
    size_t label_depth,
    bool reduction_none,
    T* output_data) {
  HIP_LONG N_D = static_cast<HIP_LONG>(count);
  HIP_LONG C = static_cast<HIP_LONG>(label_depth);
  int blocksPerGrid = (int)(ceil(static_cast<float>(N_D * C) / GridDim::maxThreadsPerBlock));
  if (reduction_none) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(_WeightedReductionNoneSoftmaxCrossEntropyLossGrad<T, Tin>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
        dY,
        log_prob,
        label,
        weight,
        normalize_factor,
        output_data,
        N_D,
        C);
  } else {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(_WeightedSoftmaxCrossEntropyLossGrad<T, Tin>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
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

#define SPECIALIZED_IMPL_SoftMaxEntropyLossGradImpl(T, Tin) \
  template void SoftmaxCrossEntropyLossGradImpl(            \
      const T* dY,                                          \
      const T* log_prob,                                    \
      const Tin* label,                                     \
      const T* weight,                                      \
      const T* normalize_factor,                            \
      size_t count,                                         \
      size_t label_depth,                                   \
      bool reducation_none,                                 \
      T* output_data);

SPECIALIZED_IMPL_SoftMaxEntropyLossGradImpl(float, int32_t)
SPECIALIZED_IMPL_SoftMaxEntropyLossGradImpl(float, int64_t)

#define SPECIALIZED_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(T, Tin) \
  template void ComputeWeightsSoftmaxCrossEntropyImpl(                 \
      const Tin* label,                                                \
      const T* weight,                                                 \
      size_t count,                                                    \
      size_t label_depth,                                              \
      int64_t ignore_index,                                            \
      T* weight_data_nd);

SPECIALIZED_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(float, int32_t)
SPECIALIZED_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(float, int64_t)

}  // namespace hip
}  // namespace onnxruntime