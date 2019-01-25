// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

class ReverseSequence : public OpKernel {
 public:
  explicit ReverseSequence(const OpKernelInfo& info) : OpKernel(info), batch_axis_(0), seq_axis_(1) {
    batch_axis_ = info.GetAttrOrDefault("batch_axis", batch_axis_);
    seq_axis_ = info.GetAttrOrDefault("seq_axis", seq_axis_);
    ORT_ENFORCE(batch_axis_ >= 0, "batch_axis must >= 0, yet got:", batch_axis_);
    ORT_ENFORCE(seq_axis_ >= 0, "seq_axis must >= 0, yet got:", seq_axis_);
    ORT_ENFORCE(seq_axis_ != batch_axis_, 
                "seq_axis must not euqal with batch_axis, yet got batch_axis:", 
                batch_axis_, ", seq_axis:", seq_axis_);
  }

  Status Compute(OpKernelContext* context) const override;

private:
  int64_t batch_axis_;
  int64_t seq_axis_;
};

}  // namespace contrib
}  // namespace onnxruntime
