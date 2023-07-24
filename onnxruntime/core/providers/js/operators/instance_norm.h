// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class InstanceNorm : public JsKernel {
 public:
  InstanceNorm(const OpKernelInfo& info) : JsKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("epsilon", &epsilon_).IsOK());

    JSEP_INIT_KERNEL_ATTRIBUTE(InstanceNormalization, ({
                                 "epsilon" : $1,
                               }),
                               static_cast<float>(epsilon_));
  }

 private:
  float epsilon_;
};

}  // namespace js
}  // namespace onnxruntime
