// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/training_ops/cpu/loss/reduction_type.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "orttraining/training_ops/cuda/loss/softmaxcrossentropy_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename TAcc, typename TLabel>
void SoftmaxCrossEntropyLossImpl(
    cudaStream_t stream,
    const T* log_prob,
    const TLabel* label,
    const T* weight,
    const TAcc* normalize_factor,
    size_t count,
    size_t label_depth,
    int64_t ignore_index,
    T* output_data);

template <typename T, typename TAcc, typename TLabel, typename TOut>
void SoftmaxCrossEntropyLossGradImpl(
    cudaStream_t stream,
    const T* dY,
    const T* log_prob,
    const TLabel* label,
    const T* weight,
    const TAcc* normalize_factor,
    const TOut* bias_data,
    size_t count,
    size_t label_depth,
    bool reduction_none,
    TOut* output_data);

template <typename T, typename TLabel, typename TOut>
void ComputeSoftmaxCrossEntropyWeightsImpl(
    cudaStream_t stream,
    const TLabel* label,
    const T* weight,
    size_t count,
    size_t label_depth,
    int64_t ignore_index,
    TOut* weight_data_nd);

template <typename T, typename TLabel, typename TOut>
class SoftmaxCrossEntropyLoss final : public LossBase {
 public:
  SoftmaxCrossEntropyLoss(const OpKernelInfo& info) : LossBase(info) {
    int64_t default_ignore_index = -1;
    info.GetAttrOrDefault<int64_t>("ignore_index", &ignore_index_, default_ignore_index);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t ignore_index_;
};

template <typename T, typename TLabel, typename TOut>
class SoftmaxCrossEntropyLossGrad final : public LossBase {
 public:
  SoftmaxCrossEntropyLossGrad(const OpKernelInfo& info) : LossBase(info) {
    int64_t default_ignore_index = -1;
    info.GetAttrOrDefault<int64_t>("ignore_index", &ignore_index_, default_ignore_index);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t ignore_index_;
};

}  // namespace cuda
}  // namespace onnxruntime
