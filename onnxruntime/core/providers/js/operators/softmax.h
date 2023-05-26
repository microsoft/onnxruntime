// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/framework/data_transfer_manager.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

namespace onnxruntime {
namespace js {
class Softmax final : public JsKernel {
 public:
  Softmax(const OpKernelInfo& info) : JsKernel(info) {
    int64_t axis = info.GetAttrOrDefault<int64_t>("axis", -1);
    JSEP_INIT_KERNEL_ATTRIBUTE(Softmax, ({
                                 "axis" : $1
                               }),
                               static_cast<int32_t>(axis));
  }
  Status Compute(OpKernelContext* context) const override;
};
}  // namespace js

}  // namespace onnxruntime
