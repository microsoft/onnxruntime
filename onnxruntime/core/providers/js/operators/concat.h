// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Concat : public JsKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : JsKernel(info), ConcatBase(info) {
    JSEP_INIT_KERNEL_ATTRIBUTE(Concat, ({"axis", $1}), axis_);
  }
};

}  // namespace js
}  // namespace onnxruntime
