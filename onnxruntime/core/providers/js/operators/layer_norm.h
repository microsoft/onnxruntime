// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class LayerNorm : public JsKernel {
 public:
  LayerNorm(const OpKernelInfo& info) : JsKernel(info) {
    info.GetAttrOrDefault<int64_t>("axis", &axis_, -1);
    info.GetAttrOrDefault<float>("epsilon", &epsilon_, 1e-05);

    JSEP_INIT_KERNEL_ATTRIBUTE(LayerNormalization, ({
                                 "axis" : Number($1),
                                 "epsilon" : Number($2),
                               }),
                               static_cast<int32_t>(axis_),
                               static_cast<float>(epsilon_));
  }

 private:
  int64_t axis_;
  float epsilon_;
};

}  // namespace js
}  // namespace onnxruntime
