// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename TI, typename TO>
class OneHotOp final : public OpKernel {
 public:
  OneHotOp(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    int64_t tmp_axis;
    if (op_kernel_info.GetAttr<int64_t>("axis", &tmp_axis).IsOK()) {
      if (tmp_axis < -1) { // as per spec it can be -1 or more
        ONNXRUNTIME_THROW("Value of axis is < -1");
      }
      axis_ = tmp_axis;
    }
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OneHotOp);

  int64_t axis_ = -1;
};
}  // namespace onnxruntime
