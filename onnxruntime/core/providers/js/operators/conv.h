// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

namespace onnxruntime {
namespace js {

template <typename T>
class Conv : public JsKernel {
 public:
  Conv(const OpKernelInfo& info) : JsKernel(info), conv_attrs_(info) {

    TensorShapeVector kernel_shape;
    if (conv_attrs_.kernel_shape_specified) {
        ORT_ENFORCE(info.GetAttrs("kernel_shape", kernel_shape).IsOK());
    }

    // currently only support Conv2D. TODO: support other
    JSEP_INIT_KERNEL_ATTRIBUTE(Conv, ({
        "format": "NHWC",
        "autopad": $1,
        "dilation0": $2,
        "dilation1": $3,
        "group": $4,
        "kernelshape0": $5,
        "kernelshape1": $6,
        "pad0": $7,
        "pad1": $8,
        "pad2": $9,
        "pad3": $10,
        "stride0": $11,
        "stride1": $12,
    }),
    static_cast<int32_t>(conv_attrs_.auto_pad),
    static_cast<int32_t>(conv_attrs_.dilations[0]),
    static_cast<int32_t>(conv_attrs_.dilations[1]),
    static_cast<int32_t>(conv_attrs_.group),
    static_cast<int32_t>(conv_attrs_.kernel_shape_specified ? kernel_shape[0] : 0),
    static_cast<int32_t>(conv_attrs_.kernel_shape_specified ? kernel_shape[1] : 0),
    static_cast<int32_t>(conv_attrs_.pads[0]),
    static_cast<int32_t>(conv_attrs_.pads[1]),
    static_cast<int32_t>(conv_attrs_.pads[2]),
    static_cast<int32_t>(conv_attrs_.pads[3]),
    static_cast<int32_t>(conv_attrs_.strides[0]),
    static_cast<int32_t>(conv_attrs_.strides[1])
    );
  }

 protected:
  ConvAttributes conv_attrs_;
};

}  // namespace js
}  // namespace onnxruntime
