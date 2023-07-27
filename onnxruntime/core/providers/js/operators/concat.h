// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Concat final : public JsKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : JsKernel(info), ConcatBase(info) {
    JSEP_INIT_KERNEL_ATTRIBUTE(Concat, ({"axis" : $1}), static_cast<int32_t>(axis_));
  }
};

}  // namespace js
}  // namespace onnxruntime
