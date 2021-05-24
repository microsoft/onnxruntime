// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

#include "gsl/gsl"

namespace onnxruntime {

class Shape final : public OpKernel {
 public:
  Shape(const OpKernelInfo& info) : OpKernel(info) {
  }

  // Takes a tensor as input and outputs an 1D int64 tensor
  // containing the shape of the input tensor.
  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const TensorShape& inputShape = input->Shape();

    size_t nDims = inputShape.NumDimensions();
    Tensor* output = context->Output(0, {gsl::narrow_cast<int64_t>(nDims)});

    inputShape.CopyDims(output->template MutableData<int64_t>(), nDims);
    return Status::OK();
  }
};
}  //namespace onnxruntime
