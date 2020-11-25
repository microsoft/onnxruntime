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
  Hardmax(const OpKernelInfo& info) : OpKernel{info} {
    const auto& node = info.node();
    opset_ = node.SinceVersion();

    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    } else {
      if (opset_ < 13) {
        axis_ = 1;  // opset-12 and below, the default axis value is 1
      } else {
        axis_ = -1;  // opset-13, the default axis value is -1
      }
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int axis_;
  int opset_;
};
}  // namespace onnxruntime
