// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename T>
class LRN : public OpKernel {
 public:
  LRN(const OpKernelInfo& info) : OpKernel(info) {
    int64_t size;
    ONNXRUNTIME_ENFORCE(info.GetAttr<int64_t>("size", &size).IsOK());
    size_ = gsl::narrow_cast<int>(size);
    ONNXRUNTIME_ENFORCE(size_ > 0);
    ONNXRUNTIME_ENFORCE(size_ % 2 == 1);
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ONNXRUNTIME_ENFORCE(alpha_ > 0.0f);
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
    ONNXRUNTIME_ENFORCE(beta_ > 0.0f);
    Status status = info.GetAttr<float>("bias", &bias_);
    if (!status.IsOK()) {
      bias_ = 1.0f;
    }
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 protected:
  float alpha_;
  float beta_;
  float bias_;
  int size_;
};
}  // namespace onnxruntime
