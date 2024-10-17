// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/unsqueeze.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
namespace webgpu {

class Unsqueeze final : public OpKernel, public UnsqueezeBase {
 public:
  explicit Unsqueeze(const OpKernelInfo& info) : OpKernel{info}, UnsqueezeBase(info) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    if (X == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Input tensor is not set");
    }
    const TensorShape& X_shape = X->Shape();

    TensorShapeVector axes;
    size_t num_inputs = context->InputCount();
    if (num_inputs == 2) {  // axes is an input
      const Tensor* axes_tensor = context->Input<Tensor>(1);
      ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
      ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 0 ||
                      axes_tensor->Shape().NumDimensions() == 1,
                  "An axes tensor must be a scalar or a vector tensor.");
      auto data_span = axes_tensor->template DataAsSpan<int64_t>();
      axes.assign(data_span.begin(), data_span.end());
    } else {
      axes.assign(axes_.begin(), axes_.end());
    }

    TensorShapeVector output_shape = ComputeOutputShape(X_shape, axes);
    Tensor* Y = context->Output(0, TensorShape(output_shape));
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();
    // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*X, *Y));
    }

    return Status::OK();
  }
};

}  // namespace webgpu
}  // namespace onnxruntime
