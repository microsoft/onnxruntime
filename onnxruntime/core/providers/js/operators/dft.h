// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class DFT final : public JsKernel {
 public:
  DFT(const OpKernelInfo& info) : JsKernel(info) {
    // opset 20 turned axis into an input (default -2); before that it's an attribute (default 1).
    int64_t axis = info.node().SinceVersion() < 20 ? info.GetAttrOrDefault<int64_t>("axis", 1) : -2;
    int64_t inverse = info.GetAttrOrDefault<int64_t>("inverse", 0);
    int64_t onesided = info.GetAttrOrDefault<int64_t>("onesided", 0);
    JSEP_INIT_KERNEL_ATTRIBUTE(DFT, ({
                                 "axis" : $1,
                                 "inverse" : $2,
                                 "onesided" : $3
                               }),
                               static_cast<int32_t>(axis),
                               static_cast<int32_t>(inverse),
                               static_cast<int32_t>(onesided));
  }
};

}  // namespace js
}  // namespace onnxruntime
