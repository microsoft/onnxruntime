// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/loss/reduction_type.h"
#include "core/util/math_cpuonly.h"

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
void ComputeShareSoftmaxCrossEntropyCPU(const int n,
                                        const int d,
                                        const Eigen::Index nd,
                                        const T* logit_data,
                                        T* shifted_logit,
                                        T* log_prob_data);

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
class SparseSoftmaxCrossEntropy final : public LossBase {
 public:
  explicit SparseSoftmaxCrossEntropy(const OpKernelInfo& info) : LossBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SparseSoftmaxCrossEntropy);
};

template <typename T>
class SparseSoftmaxCrossEntropyGrad final : public LossBase {
 public:
  explicit SparseSoftmaxCrossEntropyGrad(const OpKernelInfo& info) : LossBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SparseSoftmaxCrossEntropyGrad);
};

}  // namespace contrib
}  // namespace onnxruntime
