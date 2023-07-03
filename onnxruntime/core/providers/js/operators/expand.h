// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Expand : public JsKernel {
 public:
  Expand(const OpKernelInfo& info) : JsKernel(info) {
    JSEP_INIT_KERNEL(Expand);
  }
};

} // namespace js
} // namespace onnxruntime
