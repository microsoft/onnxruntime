// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <class T>
class CumProd final : public OpKernel {
 public:
  explicit CumProd(const OpKernelInfo& op_kernel_info);

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t exclusive_;
  int64_t reverse_;
};

namespace cumprod_op {

Status GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out);

}  // namespace cumprod_op
}  // namespace onnxruntime
