// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

template <typename T, typename U>
class LayerNorm : public JsKernel {
 public:
  LayerNorm(const OpKernelInfo& info) : JsKernel(info) {
    ORT_ENFORCE(info.GetAttr("axis", &axis_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("epsilon", &epsilon_).IsOK());

    JSEP_INIT_KERNEL_ATTRIBUTE(LayerNorm, ({
                                 "axis" : $1,
                                 "epsilon" : $2,
                               }),
                               static_cast<size_t>(axis_),
                               static_cast<float>(epsilon_));
  }
 private:
  int64_t axis_;
  float epsilon_;
};

}  // namespace js
}  // namespace onnxruntime
