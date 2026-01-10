// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/tensor/squeeze.h"
#include "core/providers/js/js_kernel.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
namespace js {

class Squeeze final : public JsKernel, public SqueezeBase {
 public:
  explicit Squeeze(const OpKernelInfo& info) : JsKernel(info), SqueezeBase(info) {}

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
      ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1,
                  "An axes tensor must be a vector tensor.");
      auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
      const auto* data = axes_tensor->Data<int64_t>();
      axes.assign(data, data + nDims);
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

}  // namespace js
}  // namespace onnxruntime
