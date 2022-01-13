// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {
/*
Given a tensor input, this operation inserts a dimension of 1 at the dimension index axis
of X's shape. The dimension index axis starts at zero; if you specify a negative number
of axis, it starts backward from the end.
*/

class ExpandDims final : public OpKernel {
 public:
  explicit ExpandDims(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* axis_tensor = context->Input<Tensor>(1);
    if (axis_tensor == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");

    ORT_ENFORCE(axis_tensor->Shape().IsScalar(), "An axis tensor must be a scalar tensor.");
    const int64_t axis = static_cast<int64_t>(axis_tensor->template Data<int32_t>()[0]);
    const Tensor* X = context->Input<Tensor>(0);
    if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    const TensorShape& X_shape = X->Shape();

    std::vector<int64_t> expanded_shape(X_shape.GetDimsAsVector());
    int64_t X_NumDims = X_shape.Size();
    ORT_ENFORCE(axis <= X_NumDims && axis >= -X_NumDims,
                "Axis must be within range [", -X_NumDims, ", ", X_NumDims, "].", " Axis is ", axis);
    if (axis >= 0) {
      expanded_shape.insert(expanded_shape.begin() + axis, 1);
    } else {
      expanded_shape.insert(expanded_shape.end() + axis + 1, 1);
    }

    Tensor* Y = context->Output(0, TensorShape(expanded_shape));
    CopyCpuTensor(X, Y);

    return Status::OK();
  }
};

}  // namespace contrib
}  // namespace onnxruntime
