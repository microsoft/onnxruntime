// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Gather : public JsKernel {
 public:
  Gather(const OpKernelInfo& info) : JsKernel(info) {
    ORT_ENFORCE(info.GetAttr("axis", &axis_).IsOK());

    JSEP_INIT_KERNEL_ATTRIBUTE(Gather, ({
                                 "axis" : Number($1),
                                 "inputIndices": Number($2),
                                 "inputShape": Number($3),
                               }),
                               static_cast<size_t>(axis_),
                               static_cast<size_t>
                               );
  }
 private:
  int64_t axis_;
};

}  // namespace js
}  // namespace onnxruntime
