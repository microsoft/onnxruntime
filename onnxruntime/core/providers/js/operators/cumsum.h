// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class CumSum final : public JsKernel {
 public:
  CumSum(const OpKernelInfo& info) : JsKernel(info) {
    // Process exclusive attribute
    int64_t exclusive = 0;
    auto status = info.GetAttr("exclusive", &exclusive);
    if (status.IsOK()) {
      if (exclusive == 1 || exclusive == 0) {
        exclusive = (exclusive == 1);
      } else {
        ORT_ENFORCE("attribute exclusive can only be 0 or 1");
      }
    }

    // Process reverse attribute
    int64_t reverse = 0;
    status = info.GetAttr("reverse", &reverse);
    if (status.IsOK()) {
      if (reverse == 1 || reverse == 0) {
        reverse = (reverse == 1);
      } else {
        ORT_ENFORCE("attribute reverse can only be 0 or 1");
      }
    }
    JSEP_INIT_KERNEL_ATTRIBUTE(CumSum, ({"exclusive" : Number($1), "reverse" : Number($2)}),
                               static_cast<int32_t>(exclusive),
                               static_cast<int32_t>(reverse));
  }
};

}  // namespace js
}  // namespace onnxruntime
