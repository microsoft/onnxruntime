// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/loss/cross_entropy.h"
#include "orttraining/training_ops/cpu/loss/reduction_type.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class BinaryCrossEntropy final : public LossBase {
 public:
  explicit BinaryCrossEntropy(const OpKernelInfo& info) : LossBase(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BinaryCrossEntropy);
};

template <typename T>
class BinaryCrossEntropyGrad final : public LossBase {
 public:
  explicit BinaryCrossEntropyGrad(const OpKernelInfo& info) : LossBase(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BinaryCrossEntropyGrad);
};


}  // namespace contrib
}  // namespace onnxruntime
