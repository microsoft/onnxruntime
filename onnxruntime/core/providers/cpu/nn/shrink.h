// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
class Shrink final : public OpKernel {
 public:
  Shrink(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    float bias_temp;
    ORT_ENFORCE(op_kernel_info.GetAttr<float>("bias", &bias_temp).IsOK());
    bias_ = gsl::narrow_cast<float>(bias_temp);

    float lambd_temp;
    ORT_ENFORCE(op_kernel_info.GetAttr<float>("lambd", &lambd_temp).IsOK());
    lambd_ = gsl::narrow_cast<float>(lambd_temp);
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  float bias_;
  float lambd_;
};
}  // namespace onnxruntime
