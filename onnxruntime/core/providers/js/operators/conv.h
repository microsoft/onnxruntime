// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

namespace onnxruntime {
namespace js {

class ConvBase : public JsKernel {
 public:
  ConvBase(const OpKernelInfo& info, bool is_channels_last, bool is_fused_conv) : JsKernel(info),
                                                                                  conv_attrs_(info),
                                                                                  w_is_const_(false) {
    TensorShapeVector kernel_shape;
    const size_t pads_vec_size = conv_attrs_.pads.size() == 0 ? 4 : conv_attrs_.pads.size();
    std::vector<int32_t> local_pads(pads_vec_size, 0);
    for (size_t i = 0; i < conv_attrs_.pads.size() && i < pads_vec_size; ++i) {
      local_pads[i] = gsl::narrow_cast<int32_t>(conv_attrs_.pads[i]);
    }

    if (conv_attrs_.kernel_shape_specified) {
      ORT_ENFORCE(info.GetAttrs("kernel_shape", kernel_shape).IsOK());
    }
    if (is_fused_conv) {
      ORT_THROW_IF_ERROR(info.GetAttr<std::string>("activation", &conv_attrs_.activation));
      ORT_ENFORCE(info.GetAttrs<float>("activation_params", activation_params_).IsOK());
    } else {
      conv_attrs_.activation = info.GetAttrOrDefault<std::string>("activation", "");
      activation_params_ = info.GetAttrsOrDefault<float>("activation_params", activation_params_);
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
                                   "pads" : $5 ? Array.from(HEAP32.subarray($6, $6 + $5)) : [],
                                   "strides" : [$7],
                                   "w_is_const" : () JS_ARROW(!!HEAP8[$9]),
                                   "activation" : UTF8ToString($10),
                                   "activation_params" : $11 ? Array.from(HEAPF32.subarray($12, $12 + $11)) : []
                                 }),
                                 static_cast<int32_t>(conv_attrs_.auto_pad),
                                 static_cast<int32_t>(conv_attrs_.dilations.size() > 0 ? conv_attrs_.dilations[0] : 0),
                                 static_cast<int32_t>(conv_attrs_.group),
                                 static_cast<int32_t>(conv_attrs_.kernel_shape_specified && kernel_shape.size() > 0 ?
                                                                                                  kernel_shape[0] : 0),
                                 static_cast<int32_t>(local_pads.size()),
                                 reinterpret_cast<int32_t>(local_pads.size() > 0 ? local_pads.data() : nullptr) >> 2,
                                 static_cast<int32_t>(conv_attrs_.strides.size() > 0 ? conv_attrs_.strides[0] : 0),
                                 static_cast<int32_t>(channels_last),
                                 reinterpret_cast<int32_t>(&w_is_const_),
                                 conv_attrs_.activation.c_str(),
                                 activation_params_.size(),
                                 reinterpret_cast<int32_t>(activation_params_.size() > 0 ?
                                                                            activation_params_.data() : nullptr) >> 2);
    } else {
      JSEP_INIT_KERNEL_ATTRIBUTE(Conv, ({
                                   "format" : $11 ? "NHWC" : "NCHW",
                                   "auto_pad" : $1,
                                   "dilations" : [ $2, $3 ],
                                   "group" : $4,
                                   "kernel_shape" : [ $5, $6 ],
                                   "pads" : $7 ? Array.from(HEAP32.subarray($8, $8 + $7)) : [],
                                   "strides" : [ $9, $10 ],
                                   "w_is_const" : () JS_ARROW(!!HEAP8[$12]),
                                   "activation" : UTF8ToString($13),
                                   "activation_params" : $14 ? Array.from(HEAPF32.subarray($15, $15 + $14)) : []
                                 }),
                                 static_cast<int32_t>(conv_attrs_.auto_pad),
                                 static_cast<int32_t>(conv_attrs_.dilations.size() > 0 ? conv_attrs_.dilations[0] : 0),
                                 static_cast<int32_t>(conv_attrs_.dilations.size() > 1 ? conv_attrs_.dilations[1] : 0),
                                 static_cast<int32_t>(conv_attrs_.group),
                                 static_cast<int32_t>(conv_attrs_.kernel_shape_specified && kernel_shape.size() > 0 ?
                                                                                                  kernel_shape[0] : 0),
                                 static_cast<int32_t>(conv_attrs_.kernel_shape_specified && kernel_shape.size() > 1 ?
                                                                                                  kernel_shape[1] : 0),
                                 static_cast<int32_t>(local_pads.size()),
                                 reinterpret_cast<int32_t>(local_pads.size() > 0 ? local_pads.data() : nullptr) >> 2,
                                 static_cast<int32_t>(conv_attrs_.strides.size() > 0 ? conv_attrs_.strides[0] : 0),
                                 static_cast<int32_t>(conv_attrs_.strides.size() > 1 ? conv_attrs_.strides[1] : 0),
                                 static_cast<int32_t>(channels_last),
                                 reinterpret_cast<int32_t>(&w_is_const_),
                                 conv_attrs_.activation.c_str(),
                                 activation_params_.size(),
                                 reinterpret_cast<int32_t>(activation_params_.size() > 0 ?
                                                                            activation_params_.data() : nullptr) >> 2);
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
  std::vector<float> activation_params_;
  // Tensor w_transposed_;
};

template <bool is_channels_last, bool is_fused_conv = false>
class Conv : public ConvBase {
 public:
  explicit Conv(const OpKernelInfo& info) : ConvBase(info, is_channels_last, is_fused_conv) {
  }
};

}  // namespace js
}  // namespace onnxruntime
