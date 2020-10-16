// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class TopKGrad final : public OpKernel {
 public:
  TopKGrad(const OpKernelInfo& info) : OpKernel(info) {
    int64_t tmp_axis;
    if (info.GetAttr<int64_t>("axis", &tmp_axis).IsOK()) {
      axis_ = tmp_axis;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  Status ComputeImpl(const Tensor& indices, const Tensor& grad, Tensor& output) const;

  int64_t axis_ = -1;
};

}  // namespace contrib
}  // namespace onnxruntime
