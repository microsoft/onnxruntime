// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

class ConcatBase {
 protected:
  ConcatBase(const OpKernelInfo& info, bool is_sequence_op = false) {
    if (!info.GetAttr("axis", &axis_).IsOK()) {
      ORT_ENFORCE(false, "Must have valid 'axis' attribute");
    }
    is_sequence_op_ = is_sequence_op;
  }

  struct Prepare {
    struct InputInfo {
      const Tensor* tensor;
      size_t num_elements;
      int64_t axis_pitch;
    };
    std::vector<InputInfo> inputs;
    int64_t output_num_elements;
    int64_t output_axis_pitch;
    Tensor* output_tensor;
    uint64_t axis;
    bool is_string_type;
  };

  Status ValidateAndConcatenateInputs(OpKernelContext* ctx, int input_count) const;

  // protected members
  int64_t axis_;

 private:
  // private methods
  Status PrepareForCompute(OpKernelContext* ctx, int input_count,
                           const std::vector<Tensor>& input_tensors, Prepare& p) const;

  Status ComputeImpl(Prepare& p) const;

  // private members
  bool is_sequence_op_;
};

class Concat final : public OpKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : OpKernel(info), ConcatBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
