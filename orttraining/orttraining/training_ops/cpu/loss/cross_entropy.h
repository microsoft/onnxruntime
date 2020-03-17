// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/loss/reduction_type.h"

namespace onnxruntime {
namespace contrib {

class LossBase : public OpKernel {
 public:
  explicit LossBase(const OpKernelInfo& info) : OpKernel(info) {
    std::string reduction;
    ORT_ENFORCE(info.GetAttr<std::string>("reduction", &reduction).IsOK());
    reduction_ = StringToReductionType(reduction);
  }

 protected:
  ReductionType reduction_;
};

template <typename T>
class SoftmaxCrossEntropy final : public LossBase {
 public:
  explicit SoftmaxCrossEntropy(const OpKernelInfo& info) : LossBase(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxCrossEntropy);
};

template <typename T>
class SoftmaxCrossEntropyGrad final : public LossBase {
 public:
  explicit SoftmaxCrossEntropyGrad(const OpKernelInfo& info) : LossBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxCrossEntropyGrad);
};

template <typename T>
class SoftmaxCrossEntropyLoss final : public LossBase {
 public:
  explicit SoftmaxCrossEntropyLoss(const OpKernelInfo& info) : LossBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxCrossEntropyLoss);
};

template <typename T>
class SoftmaxCrossEntropyLossGrad final : public LossBase {
 public:
  explicit SoftmaxCrossEntropyLossGrad(const OpKernelInfo& info) : LossBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxCrossEntropyLossGrad);
};

}  // namespace contrib
}  // namespace onnxruntime
