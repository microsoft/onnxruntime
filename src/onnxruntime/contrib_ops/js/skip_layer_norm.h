// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsKernel;

template <bool simplified>
class SkipLayerNorm final : public JsKernel {
 public:
  SkipLayerNorm(const OpKernelInfo& op_kernel_info) : JsKernel(op_kernel_info) {
    float epsilon;
    ORT_ENFORCE(op_kernel_info.GetAttr("epsilon", &epsilon).IsOK());
    ORT_ENFORCE(epsilon >= 0);
    JSEP_INIT_KERNEL_ATTRIBUTE(SkipLayerNormalization, ({
                                 "epsilon" : $1,
                                 "simplified" : !!$2
                               }),
                               epsilon,
                               static_cast<int32_t>(simplified));
  }
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
