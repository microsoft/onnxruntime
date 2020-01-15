// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
template <typename T>
class Hardmax final : public OpKernel {
 public:
  Hardmax(const OpKernelInfo& info) : OpKernel{info}, axis_{1} {
    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int axis_;
};
}  // namespace onnxruntime
