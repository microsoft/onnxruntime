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
        "autopad": $0,
        "dilation0": $1,
        "dilation1": $2,
        "group": $3,
        "kernelshape0": $4,
        "kernelshape1": $5,
        "pad0": $6,
        "pad1": $7,
        "pad2": $8,
        "pad3": $9,
        "stride0": $10,
        "stride1": $11,
    }),
    conv_attrs_.auto_pad,
    conv_attrs_.dilations[0],
    conv_attrs_.dilations[1],
    conv_attrs_.group,
    conv_attrs_.kernel_shape_specified ? kernel_shape[0] : 0,
    conv_attrs_.kernel_shape_specified ? kernel_shape[1] : 0,
    conv_attrs_.pads[0],
    conv_attrs_.pads[1],
    conv_attrs_.pads[2],
    conv_attrs_.pads[3],
    conv_attrs_.strides[0],
    conv_attrs_.strides[1]
    );
  }

 protected:
  ConvAttributes conv_attrs_;
};

}  // namespace js
}  // namespace onnxruntime
