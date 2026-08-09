// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/cann/cann_kernel.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace cann {

class Reshape final : public CannKernel {
 public:
  Reshape(const OpKernelInfo& info) : CannKernel(info) {
    allow_zero_ = (info.GetAttrOrDefault("allowzero", static_cast<int64_t>(0)) == 1);
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    const Tensor* shapeTensor = ctx->Input<Tensor>(1);
    if (shapeTensor == nullptr)
      return Status(common::ONNXRUNTIME, common::FAIL, "the 0th input is missing");
    if (shapeTensor->Shape().NumDimensions() != 1)
      return Status(common::ONNXRUNTIME, common::FAIL, "A shape tensor must be a vector tensor");

    auto data_span = shapeTensor->template DataAsSpan<int64_t>();
    TensorShapeVector shape(data_span.begin(), data_span.end());

    const Tensor* X = ctx->Input<Tensor>(0);
    if (X == nullptr)
      return Status(common::ONNXRUNTIME, common::FAIL, "the 1th input is missing");
    const TensorShape& X_shape = X->Shape();

    ReshapeHelper helper(X_shape, shape, allow_zero_);

    Tensor* Y = ctx->Output(0, TensorShape(shape));
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();
    if (target != source) {
      ORT_RETURN_IF_ERROR(CopyTensor(*X, *Y));
    }

    return Status::OK();
  }

 private:
  bool allow_zero_;
};

class Reshape_1 final : public CannKernel {
 public:
  Reshape_1(const OpKernelInfo& info) : CannKernel(info) {
    Status status = info.GetAttrs("shape", shape_);
    ORT_ENFORCE(status.IsOK(), "Attribute shape is not set.");
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    TensorShapeVector shape = shape_;
    const Tensor* X = ctx->Input<Tensor>(0);
    const TensorShape& X_shape = X->Shape();

    ReshapeHelper helper(X_shape, shape);

    Tensor* Y = ctx->Output(0, TensorShape(shape));
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();
    if (target != source) {
      ORT_RETURN_IF_ERROR(CopyTensor(*X, *Y));
    }

    return Status::OK();
  }

 private:
  TensorShapeVector shape_;
};

}  // namespace cann
}  // namespace onnxruntime
