// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
class Affine final : public OpKernel {
 public:
  Affine(const OpKernelInfo& info) : OpKernel(info) {
    // Either model-supplied or default values should be returned for alpha and beta
    ORT_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
};

template <typename T>
class Scale final : public OpKernel {
 public:
  Scale(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr("scale", &scale_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float scale_;
};

}  // namespace contrib
}  // namespace onnxruntime
