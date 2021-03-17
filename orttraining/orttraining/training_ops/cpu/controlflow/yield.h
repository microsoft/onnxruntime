// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class YieldOp final : public OpKernel {
 public:
  YieldOp(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttrs<int64_t>("full_shape_outputs", full_shape_outputs_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<int64_t> full_shape_outputs_;
};

}  // namespace contrib
}  // namespace onnxruntime
