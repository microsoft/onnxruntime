// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/framework/data_transfer_manager.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace js {

class Flatten : public JsKernel {
 public:
  Flatten(const OpKernelInfo& info) : JsKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    if (X == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    }
    const TensorShape& xShape = X->Shape();
    auto axis = axis_ >= 0 ? axis_ : HandleNegativeAxis(axis_, xShape.NumDimensions());
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(xShape.NumDimensions()) >= axis, "The rank of input tensor must be >= axis");
    const TensorShape yShape = {xShape.SizeToDimension(onnxruntime::narrow<size_t>(axis)),
                                xShape.SizeFromDimension(onnxruntime::narrow<size_t>(axis))};
    Tensor* Y = context->Output(0, yShape);
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();
    // If source and target pointers are not equal (non-inplace operation), we need to copy the data.

    if (target != source) {
      ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*X, *Y));
    }
    return Status::OK();
  }

 private:
  int64_t axis_;
};

}  // namespace js
}  // namespace onnxruntime
