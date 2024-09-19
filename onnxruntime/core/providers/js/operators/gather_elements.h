// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class GatherElements : public JsKernel {
 public:
  GatherElements(const OpKernelInfo& info) : JsKernel(info) {
    int64_t axis = info.GetAttrOrDefault<int64_t>("axis", 0);

    JSEP_INIT_KERNEL_ATTRIBUTE(GatherElements, ({
                                 "axis" : Number($1),
                               }),
                               static_cast<int32_t>(axis));
  }
};

}  // namespace js
}  // namespace onnxruntime
