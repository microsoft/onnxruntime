// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename TI, typename TO>
class OneHotOp final : public OpKernel {
 public:
  OneHot(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    int64_t tmp_axis;
    if (op_kernel_info.GetAttr<float>("axis", &tmp_axis).IsOK()) {
      axis_ = tmp_axis;
    }
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OneHotOp);

  int64_t axis_ = -1;
};
}  // namespace onnxruntime
