// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

namespace onnxruntime {
namespace js {

class ConvBase : public JsKernel {
 public:
  ConvBase(const OpKernelInfo& info, bool is_channels_last, bool is_fused_conv) : JsKernel(info),
                                                                                  conv_attrs_(info),
                                                                                  w_is_const_(false) {
    const size_t pads_vec_size = conv_attrs_.pads.size() == 0 ? 4 : conv_attrs_.pads.size();
    std::vector<int32_t> local_pads(pads_vec_size, 0);
    for (size_t i = 0; i < conv_attrs_.pads.size() && i < pads_vec_size; ++i) {
      local_pads[i] = gsl::narrow_cast<int32_t>(conv_attrs_.pads[i]);
    }

    TensorShapeVector kernel_shape;
    if (conv_attrs_.kernel_shape_specified) {
      ORT_ENFORCE(info.GetAttrs("kernel_shape", kernel_shape).IsOK());
    }
    std::vector<int32_t> kernel_shapes(kernel_shape.size(), 0);
    if (conv_attrs_.kernel_shape_specified) {
      for (size_t i = 0; i < kernel_shape.size(); ++i) {
        kernel_shapes[i] = gsl::narrow_cast<int32_t>(kernel_shape[i]);
      }
    }

    std::vector<int32_t> strides(conv_attrs_.strides.size(), 0);
    for (size_t i = 0; i < conv_attrs_.strides.size(); ++i) {
      strides[i] = gsl::narrow_cast<int32_t>(conv_attrs_.strides[i]);
    }

    std::vector<int32_t> dilations(conv_attrs_.dilations.size(), 0);
    for (size_t i = 0; i < conv_attrs_.dilations.size(); ++i) {
      dilations[i] = gsl::narrow_cast<int32_t>(conv_attrs_.dilations[i]);
    }

    conv_attrs_.activation = info.GetAttrOrDefault<std::string>("activation", "");
    std::vector<float> activation_params = info.GetAttrsOrDefault<float>("activation_params");
    int64_t channels_last = is_channels_last ? 1 : info.GetAttrOrDefault<int64_t>("channels_last", 0);

    JSEP_INIT_KERNEL_ATTRIBUTE(Conv, ({
                                 "format" : $11 ? "NHWC" : "NCHW",
                                 "auto_pad" : $1,
                                 "dilations" : $2 ? Array.from(HEAP32.subarray(Number($2), Number($3))) : [],
                                 "group" : $4,
                                 "kernel_shape" : $5 ? Array.from(HEAP32.subarray(Number($5), Number($6))) : [],
                                 "pads" : $7 ? Array.from(HEAP32.subarray(Number($7), Number($8))) : [],
                                 "strides" : $9 ? Array.from(HEAP32.subarray(Number($9), Number($10))) : [],
                                 "w_is_const" : () JS_ARROW(!!HEAP8[Number($12)]),
                                 "activation" : UTF8ToString($13),
                                 "activation_params" : $14 ? Array.from(HEAPF32.subarray(Number($14), Number($15))) : []
                               }),
                               static_cast<int32_t>(conv_attrs_.auto_pad),
                               JSEP_HEAP32_INDEX_START(dilations),
                               JSEP_HEAP32_INDEX_END(dilations),
                               static_cast<int32_t>(conv_attrs_.group),
                               JSEP_HEAP32_INDEX_START(kernel_shapes),
                               JSEP_HEAP32_INDEX_END(kernel_shapes),
                               JSEP_HEAP32_INDEX_START(local_pads),
                               JSEP_HEAP32_INDEX_END(local_pads),
                               JSEP_HEAP32_INDEX_START(strides),
                               JSEP_HEAP32_INDEX_END(strides),
                               static_cast<int32_t>(channels_last),
                               JSEP_HEAP8_INDEX(&w_is_const_),
                               conv_attrs_.activation.c_str(),
                               JSEP_HEAP32_INDEX_START(activation_params),
                               JSEP_HEAP32_INDEX_END(activation_params));
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

template <bool is_channels_last, bool is_fused_conv = false>
class Conv : public ConvBase {
 public:
  explicit Conv(const OpKernelInfo& info) : ConvBase(info, is_channels_last, is_fused_conv) {
  }
};

}  // namespace js
}  // namespace onnxruntime
