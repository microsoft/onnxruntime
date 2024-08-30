// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Einsum final : public JsKernel {
 public:
  Einsum(const OpKernelInfo& info) : JsKernel(info) {
    std::string equation;
    ORT_ENFORCE(info.GetAttr<std::string>("equation", &equation).IsOK(),
                "Missing 'equation' attribute");
    JSEP_INIT_KERNEL_ATTRIBUTE(Einsum, ({
                                 "equation" : UTF8ToString($1),
                               }),
                               equation.c_str());
  }
};

}  // namespace js
}  // namespace onnxruntime
