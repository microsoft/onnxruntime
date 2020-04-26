// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/loss/cross_entropy.h"
#include "orttraining/training_ops/cpu/loss/reduction_type.h"

namespace onnxruntime {
namespace contrib {

template <typename T1, typename T2>
class SoftmaxCrossEntropyLoss final : public LossBase {
 public:
  explicit SoftmaxCrossEntropyLoss(const OpKernelInfo& info) : LossBase(info) {
    int64_t default_ignore_index = -1;
    info.GetAttrOrDefault<int64_t>("ignore_index", &ignore_index_, default_ignore_index);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t ignore_index_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxCrossEntropyLoss);
};

template <typename T1, typename T2>
class SoftmaxCrossEntropyLossGrad final : public LossBase {
 public:
  explicit SoftmaxCrossEntropyLossGrad(const OpKernelInfo& info) : LossBase(info) {
    int64_t default_ignore_index = -1;
    info.GetAttrOrDefault<int64_t>("ignore_index", &ignore_index_, default_ignore_index);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t ignore_index_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxCrossEntropyLossGrad);
};

void VerifyLogitWeightAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape,
                                    const TensorShape* weight_shape);

void GetNDCFromLogitAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, int64_t& N_D, int64_t& C);
void GetPermutationAndShape(bool ncd_to_ndc, const TensorShape& tensor_shape, std::vector<int64_t>& new_shape,
                            std::vector<size_t>& permutations);

}  // namespace contrib
}  // namespace onnxruntime
