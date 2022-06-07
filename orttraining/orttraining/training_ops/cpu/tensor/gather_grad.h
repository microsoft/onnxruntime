// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

class GatherGrad final : public OpKernel {
 public:
  GatherGrad(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T, typename Tind>
  Status ComputeImpl(const TensorShape& data_shape, const Tensor& indices, const Tensor& grad, Tensor& output,
      concurrency::ThreadPool* tp) const;

  int64_t axis_;
};

}  // namespace contrib
}  // namespace onnxruntime
