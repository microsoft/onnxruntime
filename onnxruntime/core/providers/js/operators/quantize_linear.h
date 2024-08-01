// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class DequantizeLinear : public JsKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : JsKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
    if (!info.GetAttr<int64_t>("block_size", &axis_).IsOK()) {
      block_size_ = 0;
    }
    JSEP_INIT_KERNEL_ATTRIBUTE(DequantizeLinear, ({
                                 "axis" : $1,
                                 "blockSize" : $2
                               }),
                               static_cast<int32_t>(axis_), static_cast<int32_t>(block_size_));
  }

 private:
  int64_t axis_;
  int64_t block_size_;
};

}  // namespace js
}  // namespace onnxruntime
