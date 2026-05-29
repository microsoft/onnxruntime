// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"

#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {

#ifndef SHARED_PROVIDER
template <class T>
class CumSum final : public OpKernel {
 public:
  explicit CumSum(const OpKernelInfo& op_kernel_info);

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t exclusive_;
  int64_t reverse_;
};
#endif

namespace cumsum_op {

#ifdef SHARED_PROVIDER
Status GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out);
#else
inline Status GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) {
  if (!axis_tensor)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor must be provided to the CumSum op");

  if (axis_tensor->Shape().NumDimensions() > 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor should be 0D or 1D");

  if (axis_tensor->Shape().Size() != 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor must contain exactly one element");

  if (axis_tensor->IsDataType<int32_t>()) {
    axis_out = static_cast<int64_t>(axis_tensor->Data<int32_t>()[0]);
  } else if (axis_tensor->IsDataType<int64_t>()) {
    axis_out = axis_tensor->Data<int64_t>()[0];
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor should be of type `int32_t` or `int64_t`");
  }

  axis_out = HandleNegativeAxis(axis_out, input_rank);

  return Status::OK();
}
#endif  // SHARED_PROVIDER

}  // namespace cumsum_op
}  // namespace onnxruntime
