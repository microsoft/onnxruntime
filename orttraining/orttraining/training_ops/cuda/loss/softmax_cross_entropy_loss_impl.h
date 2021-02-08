// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/training_ops/cpu/loss/reduction_type.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "orttraining/training_ops/cuda/loss/softmaxcrossentropy_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename Tin>
void SoftmaxCrossEntropyLossImpl(
    cudaStream_t stream,
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    size_t count,
    size_t label_depth,
    int64_t ignore_index,
    T* output_data);

template <typename T, typename Tin>
void SoftmaxCrossEntropyLossGradImpl(
    cudaStream_t stream,
    const T* dY,
    const T* log_prob,
    const Tin* label,
    const T* weight,
    const T* normalize_factor,
    size_t count,
    size_t label_depth,
    bool reduction_none,
    T* output_data);

template <typename T, typename Tin>
void ComputeWeightsSoftmaxCrossEntropyImpl(
    cudaStream_t stream,
    const Tin* label,
    const T* weight,
    size_t count,
    size_t label_depth,
    int64_t ignore_index,
    T* weight_data_nd);

template <typename T, typename Tin>
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

template <typename T, typename Tin>
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
