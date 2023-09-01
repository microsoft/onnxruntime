// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/framework/data_transfer_manager.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

namespace onnxruntime {
namespace js {

class Reshape final : public JsKernel {
 public:
  Reshape(const OpKernelInfo& info) : JsKernel(info),
                                      allow_zero_(info.GetAttrOrDefault("allowzero", static_cast<int64_t>(0)) == 1) {
  }

  Status Compute(OpKernelContext* context) const override {
    // Copy the second input tensor into the shape vector
    const Tensor* shapeTensor = context->Input<Tensor>(1);
    if (shapeTensor == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    if (shapeTensor->Shape().NumDimensions() != 1) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "A shape tensor must be a vector tensor, got ", shapeTensor->Shape().NumDimensions(), " dimensions");
    auto data_span = shapeTensor->template DataAsSpan<int64_t>();
    TensorShapeVector shape(data_span.begin(), data_span.end());
    const Tensor* X = context->Input<Tensor>(0);
    if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    const TensorShape& X_shape = X->Shape();

    ReshapeHelper helper(X_shape, shape, allow_zero_);

    Tensor* Y = context->Output(0, TensorShape(shape));
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();
    // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*X, *Y));
    }

    return Status::OK();
  }

 private:
  bool allow_zero_;
};

}  // namespace js
}  // namespace onnxruntime
