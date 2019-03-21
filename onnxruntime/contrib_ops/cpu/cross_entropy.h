// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class SoftmaxCrossEntropy final : public OpKernel {
 public:
  explicit SoftmaxCrossEntropy(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxCrossEntropy);
};

template <typename T>
class SoftmaxCrossEntropyGrad final : public OpKernel {
 public:
  explicit SoftmaxCrossEntropyGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxCrossEntropyGrad);
};

}  // namespace contrib
}  // namespace onnxruntime
