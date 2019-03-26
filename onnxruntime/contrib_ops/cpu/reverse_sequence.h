// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

class ReverseSequenceOp : public OpKernel {
 public:
  explicit ReverseSequenceOp(const OpKernelInfo& info) : OpKernel(info) {
    //std::string data_format;
    //ORT_ENFORCE(info.GetAttr<std::string>("data_format", &data_format).IsOK());

    //if (data_format == "time_major")
    //  time_major_ = true;
    //else if (data_format == "batch_major")
    //  time_major_ = false;
    //else
    //  ORT_THROW("Invalid 'data_format' argument of '", data_format,
    //            "'. Must be one of 'time_major' or 'batch_major'.");

    int64_t batch_axis, seq_axis;
    ORT_ENFORCE(info.GetAttr<int64_t>("batch_axis", &batch_axis).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("seq_axis", &seq_axis).IsOK());

    ORT_ENFORCE(batch_axis < 2, "batch_axis must be 0 or 1. Got:", batch_axis);
    ORT_ENFORCE(seq_axis < 2, "seq_axis must be 0 or 1. Got:", seq_axis);
    ORT_ENFORCE(batch_axis != seq_axis, "batch_axis and seq_axis can not have the same value.");

    time_major_ = seq_axis == 0;
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool time_major_;
};

}  // namespace contrib
}  // namespace onnxruntime
