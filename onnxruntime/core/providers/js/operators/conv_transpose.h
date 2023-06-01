// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/common/gsl.h"
#include "core/providers/cpu/nn/conv_transpose_attributes.h"

namespace onnxruntime {
namespace js {
template <typename T, bool is_channels_last>
class ConvTranspose : public JsKernel {
 public:
  ConvTranspose(const OpKernelInfo& info) : JsKernel(info), conv_transpose_attrs_(info), w_is_const_(false) {
    TensorShapeVector kernel_shape;
    if (conv_transpose_attrs_.kernel_shape_specified) {
      ORT_ENFORCE(info.GetAttrs("kernel_shape", kernel_shape).IsOK());
    }

    int64_t channels_last = is_channels_last ? 1 : info.GetAttrOrDefault<int64_t>("channels_last", 0);

    // currently only support Conv 1D/2D. TODO: support Conv3D and other
    if (conv_transpose_attrs_.dilations.size() == 1 ||
        (conv_transpose_attrs_.kernel_shape_specified && kernel_shape.size() == 1) ||
        conv_transpose_attrs_.strides.size() == 1) {
      JSEP_INIT_KERNEL_ATTRIBUTE(ConvTranspose, ({
                                   "format" : $8 ? "NHWC" : "NCHW",
                                   "autoPad" : $1,
                                   "dilations" : [$2],
                                   "group" : $3,
                                   "kernel_shape" : [$4],
                                   "pads" : [ $5, $6 ],
                                   "strides" : [$7],
                                   "wIsConst" : () JS_ARROW(!!HEAP8[$9]),
                                   "output_padding" : $10 ? Array.from(HEAP32.subarray($11, $11 + $10)) : [],
                                   "output_shape" : $12 ? Array.from(HEAP32.subarray($12, $13 + $12)) : []
                                 }),
                                 static_cast<int32_t>(conv_transpose_attrs_.auto_pad),
                                 static_cast<int32_t>(conv_transpose_attrs_.dilations.size() > 0 ? conv_transpose_attrs_.dilations[0] : 0),
                                 static_cast<int32_t>(conv_transpose_attrs_.group),
                                 static_cast<int32_t>(conv_transpose_attrs_.kernel_shape_specified && kernel_shape.size() > 0) ? kernel_shape[0] : 0,
                                 static_cast<int32_t>(conv_transpose_attrs_.pads.size()),
                                 static_cast<int32_t>(conv_transpose_attrs_.pads.size() > 1) ? conv_transpose_attrs_.pads[1] : 0,
                                 static_cast<int32_t>(conv_transpose_attrs_.strides.size() > 0) ? conv_transpose_attrs_.strides[0] : 0,
                                 static_cast<int32_t>(channels_last),
                                 reinterpret_cast<int32_t>(&w_is_const_),
                                 gsl::narrow_cast<int32_t>(conv_transpose_attrs_.output_shape.size()),
                                 reinterpret_cast<int32_t>(conv_transpose_attrs_.output_padding.size() > 0 ? conv_transpose_attrs_.output_padding.data() : nullptr) >> 2,
                                 gsl::narrow_cast<int32_t>(conv_transpose_attrs_.output_shape.size()),
                                 reinterpret_cast<int32_t>(conv_transpose_attrs_.output_shape.size() > 0 ? conv_transpose_attrs_.output_shape.data() : nullptr) >> 2);
    } else {
      JSEP_INIT_KERNEL_ATTRIBUTE(ConvTranspose, ({
                                   "format" : $13 ? "NHWC" : "NCHW",
                                   "autoPad" : $1,
                                   "dilations" : [ $2, $3 ],
                                   "group" : $4,
                                   "kernelShape" : [ $5, $6 ],
                                   "pads" : [ $7, $8, $9, $10 ],
                                   "strides" : [ $11, $12 ],
                                   "wIsConst" : () JS_ARROW(!!HEAP8[$14]),
                                   "outputPadding" : ($15 > 0) ? Array.from(HEAP32.subarray($16, $16 + $15)) : [],
                                   "outputShape" : ($17 > 0) ? Array.from(HEAP32.subarray($18, $18 + $17)) : []
                                 }),
                                 static_cast<int32_t>(conv_transpose_attrs_.auto_pad),
                                 static_cast<int32_t>(conv_transpose_attrs_.dilations.size() > 0 ? conv_transpose_attrs_.dilations[0] : 0),
                                 static_cast<int32_t>(conv_transpose_attrs_.dilations.size() > 1 ? conv_transpose_attrs_.dilations[1] : 0),
                                 static_cast<int32_t>(conv_transpose_attrs_.group),
                                 static_cast<int32_t>(conv_transpose_attrs_.kernel_shape_specified && kernel_shape.size() > 0 ? kernel_shape[0] : 0),
                                 static_cast<int32_t>(conv_transpose_attrs_.kernel_shape_specified && kernel_shape.size() > 1 ? kernel_shape[1] : 0),
                                 static_cast<int32_t>(conv_transpose_attrs_.pads.size() > 0 ? conv_transpose_attrs_.pads[0] : 0),
                                 static_cast<int32_t>(conv_transpose_attrs_.pads.size() > 1 ? conv_transpose_attrs_.pads[1] : 0),
                                 static_cast<int32_t>(conv_transpose_attrs_.pads.size() > 2 ? conv_transpose_attrs_.pads[2] : 0),
                                 static_cast<int32_t>(conv_transpose_attrs_.pads.size() > 3 ? conv_transpose_attrs_.pads[3] : 0),
                                 static_cast<int32_t>(conv_transpose_attrs_.strides.size() > 0 ? conv_transpose_attrs_.strides[0] : 0),
                                 static_cast<int32_t>(conv_transpose_attrs_.strides.size() > 1 ? conv_transpose_attrs_.strides[1] : 0),
                                 static_cast<int32_t>(channels_last),
                                 reinterpret_cast<int32_t>(&w_is_const_),
                                 gsl::narrow_cast<int32_t>(conv_transpose_attrs_.output_shape.size()),
                                 reinterpret_cast<int32_t>(conv_transpose_attrs_.output_padding.size() > 0 ? conv_transpose_attrs_.output_padding.data() : nullptr) >> 2,
                                 gsl::narrow_cast<int32_t>(conv_transpose_attrs_.output_shape.size()),
                                 reinterpret_cast<int32_t>(conv_transpose_attrs_.output_shape.size() > 0 ? conv_transpose_attrs_.output_shape.data() : nullptr) >> 2);
    }
  }

 protected:
  ConvTransposeAttributes conv_transpose_attrs_;
  bool w_is_const_;
};

}  // namespace js
}  // namespace onnxruntime
