// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class BatchNormalizationGrad final : public OpKernel {
 public:
  explicit BatchNormalizationGrad(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BatchNormalizationGrad);
};

}  // namespace contrib
}  // namespace onnxruntime
