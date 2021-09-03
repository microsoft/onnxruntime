// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

#include "gsl/gsl"
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

    if (!needs_slicing_) {  // vanilla use of Shape (no slicing)
      Tensor* output = context->Output(0, {rank});
      input_shape.CopyDims(output->template MutableData<int64_t>(), static_cast<size_t>(rank));
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
        input_shape.CopyDims(output->template MutableData<int64_t>(), true_start, slice_length);
      }
    }

    return Status::OK();
  }

 private:
  bool needs_slicing_ = false;
  int64_t start_index_ = 0;
  int64_t end_index_ = std::numeric_limits<int64_t>::max();
};

class TransposeOfShape final : public OpKernel {
 public:
  TransposeOfShape(const OpKernelInfo& info) : OpKernel(info) {
    std::vector<int64_t> temp_perm;
    Status status = info.GetAttrs<int64_t>("perm", temp_perm);
    if (status.IsOK()) {
      size_t rank = temp_perm.size();
      perm_.resize(temp_perm.size());
      // Check that perm_ is a valid permutation of [0,rank-1]
      for (size_t i = 0; i != temp_perm.size(); ++i) {
        int64_t v = temp_perm[i];
        ORT_ENFORCE(v >= 0 && static_cast<uint64_t>(v) <= std::numeric_limits<size_t>::max());
        if (static_cast<size_t>(v) >= rank)
          ORT_THROW("Attribute perm of TransposeOfShape has an invalid value. Value ", i, " is outside range.");
        perm_[i] = static_cast<size_t>(v);
      }
      perm_specified_ = true;
      std::vector<bool> seen(rank, false);
      for (auto i : perm_) {
        if (seen[i])
          ORT_THROW("Attribute perm of TransposeOfShape has an invalid value. Value ", i, " is repeated.");
        seen[i] = true;
      }
    }
  }

  // Takes a tensor as input and outputs an 1D int64 tensor
  // containing the shape of the input tensor.
  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const TensorShape& input_shape = input->Shape();

    size_t rank = input_shape.NumDimensions();
    Tensor* output = context->Output(0, {gsl::narrow_cast<int64_t>(rank)});
    int64_t* output_data = output->template MutableData<int64_t>();

    if (perm_specified_) {  // vanilla use of Shape (no slicing)
      for (size_t i = 0; i < rank; ++i) {
        output_data[i] = input_shape[perm_[i]];
      }
    } else {  
      for (size_t i = 0; i < rank; ++i) {
        output_data[i] = input_shape[rank-i-1];
      }
    }

    return Status::OK();
  }

 private:
  std::vector<size_t> perm_;
  bool perm_specified_ = false;
};

}  //namespace onnxruntime
