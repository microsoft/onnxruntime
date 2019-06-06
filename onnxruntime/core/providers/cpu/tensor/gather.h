// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {

class GatherBase {
 protected:
  GatherBase(const OpKernelInfo& info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }

  struct Prepare {
    const Tensor* input_tensor;
    const Tensor* indices_tensor;
    Tensor* output_tensor;
    int64_t axis;
  };

  Status PrepareForCompute(OpKernelContext* context, Prepare& p) const;

 private:
  int64_t axis_;
};

class Gather final : public OpKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : OpKernel(info), GatherBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

namespace contrib {
class GatherGrad final : public OpKernel {
 public:
  GatherGrad(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};
}  // namespace contrib

}  // namespace onnxruntime
