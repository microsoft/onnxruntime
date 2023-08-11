// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

namespace onnxruntime {
namespace js {
template <typename T>
class Softmax : public JsKernel {
 public:
  Softmax(const OpKernelInfo& info) : JsKernel(info) {
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
    JSEP_INIT_KERNEL_ATTRIBUTE(Softmax, ({
                                 "axis" : $1
                               }),
                               axis_);
  }

 private:
  int axis_;
  int opset_;
};

}  // namespace js
}  // namespace onnxruntime
