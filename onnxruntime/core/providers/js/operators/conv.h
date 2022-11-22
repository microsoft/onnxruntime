// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

namespace onnxruntime {
namespace js {

template <typename T, bool is_channels_last>
class Conv : public JsKernel {
 public:
  Conv(const OpKernelInfo& info) : JsKernel(info), conv_attrs_(info) {

    TensorShapeVector kernel_shape;
    if (conv_attrs_.kernel_shape_specified) {
        ORT_ENFORCE(info.GetAttrs("kernel_shape", kernel_shape).IsOK());
    }

    // currently only support Conv2D. TODO: support other
    JSEP_INIT_KERNEL_ATTRIBUTE(Conv, ({
        "format": $13 ? "NHWC" : "NCHW",
        "auto_pad": $1,
        "dilations": [$2, $3],
        "group": $4,
        "kernel_shape": [$5, $6],
        "pads": [$7, $8, $9, $10],
        "strides": [$11, $12]
    }),
    static_cast<int32_t>(conv_attrs_.auto_pad),
    static_cast<int32_t>(conv_attrs_.dilations.size() > 0 ? conv_attrs_.dilations[0] : 0),
    static_cast<int32_t>(conv_attrs_.dilations.size() > 1 ? conv_attrs_.dilations[1] : 0),
    static_cast<int32_t>(conv_attrs_.group),
    static_cast<int32_t>(conv_attrs_.kernel_shape_specified && kernel_shape.size() > 0 ? kernel_shape[0] : 0),
    static_cast<int32_t>(conv_attrs_.kernel_shape_specified && kernel_shape.size() > 1 ? kernel_shape[1] : 0),
    static_cast<int32_t>(conv_attrs_.pads.size() > 0 ? conv_attrs_.pads[0] : 0),
    static_cast<int32_t>(conv_attrs_.pads.size() > 1 ? conv_attrs_.pads[1] : 0),
    static_cast<int32_t>(conv_attrs_.pads.size() > 2 ? conv_attrs_.pads[2] : 0),
    static_cast<int32_t>(conv_attrs_.pads.size() > 3 ? conv_attrs_.pads[3] : 0),
    static_cast<int32_t>(conv_attrs_.strides.size() > 0 ? conv_attrs_.strides[0] : 0),
    static_cast<int32_t>(conv_attrs_.strides.size() > 1 ? conv_attrs_.strides[1] : 0),
    static_cast<int32_t>(is_channels_last)
    );
  }

 protected:
  ConvAttributes conv_attrs_;
};

}  // namespace js
}  // namespace onnxruntime
