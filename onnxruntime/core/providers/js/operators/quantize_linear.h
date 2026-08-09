// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class DequantizeLinear : public JsKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : JsKernel(info) {
    int64_t axis;
    int64_t block_size;
    if (!info.GetAttr<int64_t>("axis", &axis).IsOK()) {
      axis = 1;
    }
    if (!info.GetAttr<int64_t>("block_size", &block_size).IsOK()) {
      block_size = 0;
    }
    JSEP_INIT_KERNEL_ATTRIBUTE(DequantizeLinear, ({
                                 "axis" : $1,
                                 "blockSize" : $2
                               }),
                               static_cast<int32_t>(axis), static_cast<int32_t>(block_size));
  }
};

}  // namespace js
}  // namespace onnxruntime
