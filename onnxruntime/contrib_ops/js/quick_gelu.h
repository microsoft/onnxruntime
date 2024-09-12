// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsKernel;

class QuickGelu final : public JsKernel {
 public:
  explicit QuickGelu(const OpKernelInfo& info) : JsKernel(info) {
    float alpha = info.GetAttrOrDefault<float>("alpha", 1.0);
    JSEP_INIT_KERNEL_ATTRIBUTE(QuickGelu, ({"alpha" : $1}), alpha);
  }
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
