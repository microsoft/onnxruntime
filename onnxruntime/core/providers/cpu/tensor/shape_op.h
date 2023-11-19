// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#endif

#include "core/common/gsl.h"
#include <limits>

namespace onnxruntime {

class Shape final : public OpKernel {
 public:
  Shape(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<int64_t>("start", &start_index_, 0);

    if (start_index_ != 0) {
      // "start" is provided and is non-default (default is 0)
      needs_slicing_ = true;
    }

    if (info.GetAttr<int64_t>("end", &end_index_).IsOK()) {
      needs_slicing_ = true;
    }
  }

  // Takes a tensor as input and outputs an 1D int64 tensor
  // containing the shape of the input tensor.
  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const TensorShape& input_shape = input->Shape();

    int64_t rank = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());
    // ONNX shape inferencing doesn't work with a scalar. Spec does not say it's unsupported.
    ORT_ENFORCE(rank != 0, "Shape of a scalar is not supported");

    if (!needs_slicing_) {  // vanilla use of Shape (no slicing)
      Tensor* output = context->Output(0, {rank});
      input_shape.CopyDims(output->MutableData<int64_t>(), static_cast<size_t>(rank));
    } else {  // slicing is needed
      int64_t true_start = start_index_;
      int64_t true_end = end_index_;

      // Deal with negative(s) and clamp
      true_start = true_start < 0 ? true_start + rank : true_start;
      true_start = true_start < 0 ? 0 : ((true_start > rank) ? rank : true_start);

      true_end = true_end < 0 ? true_end + rank : true_end;
      true_end = true_end < 0 ? 0 : ((true_end > rank) ? rank : true_end);

      auto slice_length = true_end - true_start;
      Tensor* output = context->Output(0, {slice_length < 0 ? 0 : slice_length});

      if (slice_length > 0) {
        input_shape.CopyDims(output->MutableData<int64_t>(), onnxruntime::narrow<size_t>(true_start), onnxruntime::narrow<size_t>(slice_length));
      }
    }

    return Status::OK();
  }

 private:
  bool needs_slicing_ = false;
  int64_t start_index_ = 0;
  int64_t end_index_ = std::numeric_limits<int64_t>::max();
};

}  // namespace onnxruntime
