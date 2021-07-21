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
    info.GetAttrOrDefault<int64_t>("start", &start_index_, 0);

    if (start_index_ > 0) {
      needs_slicing_ = true;
    }

    if (info.GetAttr<int64_t>("end", &end_index_).IsOK()) {
      // "end" is provided
      end_provided_ = true;
      needs_slicing_ = true;
    }
  }

  // Takes a tensor as input and outputs an 1D int64 tensor
  // containing the shape of the input tensor.
  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const TensorShape& input_shape = input->Shape();

    size_t rank = input_shape.NumDimensions();

    if (!needs_slicing_) {  // vanilla use of Shape (no slicing)
      Tensor* output = context->Output(0, {gsl::narrow_cast<int64_t>(rank)});
      input_shape.CopyDims(output->template MutableData<int64_t>(), rank);
    } else {  // slicing is needed
      int64_t true_start = start_index_;

      if (true_start < 0) {
        true_start += rank;
      }

      true_start = true_start < 0
                       ? 0
                       : ((true_start > rank) ? rank : true_start);

      int64_t true_end = rank;

      if (end_provided_) {  // end was explicitly provided, so honor that
        true_end = end_index_;

        if (true_end < 0) {
          true_end += rank;
        }

        true_end = true_end < 0
                       ? 0
                       : ((true_end > rank) ? rank : true_end);
      }

      auto slice_size = true_end - true_start;
      Tensor* output = context->Output(0, {slice_size < 0 ? 0 : slice_size});

      if (slice_size > 0) {
        input_shape.CopyDims(output->template MutableData<int64_t>(), true_start, slice_size);
      }
    }

    return Status::OK();
  }

 private:
  bool needs_slicing_ = false;
  bool end_provided_ = false;
  int64_t start_index_ = 0;
  int64_t end_index_ = -1;  // only relevant if `end_provided_` is true
};

}  //namespace onnxruntime
