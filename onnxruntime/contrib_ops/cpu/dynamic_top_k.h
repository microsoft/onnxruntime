// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
class DynamicTopK final : public OpKernel {
 public:
  DynamicTopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    int64_t axis_temp;
    ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
    axis_ = gsl::narrow_cast<int>(axis_temp);
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  int axis_;
};
}  // namespace contrib
}  // namespace onnxruntime
