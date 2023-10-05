// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

namespace onnxruntime {
namespace js {

template <bool is_channels_last, bool has_activation = false>
class Conv : public JsKernel {
 public:
  Conv(const OpKernelInfo& info) : JsKernel(info), conv_attrs_(info), w_is_const_(false) {
    TensorShapeVector kernel_shape;
    if (conv_attrs_.kernel_shape_specified) {
      ORT_ENFORCE(info.GetAttrs("kernel_shape", kernel_shape).IsOK());
    }
    if (has_activation) {
      ORT_THROW_IF_ERROR(info.GetAttr<std::string>("activation", &conv_attrs_.activation));
    } else {
      conv_attrs_.activation = info.GetAttrOrDefault<std::string>("activation", "");
    }

    int64_t channels_last = is_channels_last ? 1 : info.GetAttrOrDefault<int64_t>("channels_last", 0);

    // currently only support Conv 1D/2D. TODO: support Conv3D and other
    if (conv_attrs_.dilations.size() == 1 ||
        (conv_attrs_.kernel_shape_specified && kernel_shape.size() == 1) ||
        conv_attrs_.strides.size() == 1) {
      JSEP_INIT_KERNEL_ATTRIBUTE(Conv, ({
                                   "format" : $8 ? "NHWC" : "NCHW",
                                   "auto_pad" : $1,
                                   "dilations" : [$2],
                                   "group" : $3,
                                   "kernel_shape" : [$4],
                                   "pads" : [ $5, $6 ],
                                   "strides" : [$7],
                                   "w_is_const" : () JS_ARROW(!!HEAP8[$9]),
                                   "activation" : UTF8ToString($10)
                                 }),
                                 static_cast<int32_t>(conv_attrs_.auto_pad),
                                 static_cast<int32_t>(conv_attrs_.dilations.size() > 0 ? conv_attrs_.dilations[0] : 0),
                                 static_cast<int32_t>(conv_attrs_.group),
                                 static_cast<int32_t>(conv_attrs_.kernel_shape_specified && kernel_shape.size() > 0 ? kernel_shape[0] : 0),
                                 static_cast<int32_t>(conv_attrs_.pads.size() > 0 ? conv_attrs_.pads[0] : 0),
                                 static_cast<int32_t>(conv_attrs_.pads.size() > 1 ? conv_attrs_.pads[1] : 0),
                                 static_cast<int32_t>(conv_attrs_.strides.size() > 0 ? conv_attrs_.strides[0] : 0),
                                 static_cast<int32_t>(channels_last),
                                 reinterpret_cast<int32_t>(&w_is_const_),
                                 conv_attrs_.activation.c_str());
    } else {
      JSEP_INIT_KERNEL_ATTRIBUTE(Conv, ({
                                   "format" : $13 ? "NHWC" : "NCHW",
                                   "auto_pad" : $1,
                                   "dilations" : [ $2, $3 ],
                                   "group" : $4,
                                   "kernel_shape" : [ $5, $6 ],
                                   "pads" : [ $7, $8, $9, $10 ],
                                   "strides" : [ $11, $12 ],
                                   "w_is_const" : () JS_ARROW(!!HEAP8[$14]),
                                   "activation" : UTF8ToString($15)
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
                                 static_cast<int32_t>(channels_last),
                                 reinterpret_cast<int32_t>(&w_is_const_),
                                 conv_attrs_.activation.c_str());
    }
  }

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* /* prepacked_weights */) override {
    is_packed = false;

    if (input_idx == 1) {
      // Only handle the common case of conv2D
      if (tensor.Shape().NumDimensions() != 4 || tensor.SizeInBytes() == 0) {
        return Status::OK();
      }

      w_is_const_ = true;
    }

    return Status::OK();
  }

 protected:
  ConvAttributes conv_attrs_;
  bool w_is_const_;
  // Tensor w_transposed_;
};

}  // namespace js
}  // namespace onnxruntime
