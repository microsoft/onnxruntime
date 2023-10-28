// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include "core/common/gsl.h"
#include "core/providers/cpu/nn/conv_transpose_attributes.h"
#include "core/providers/js/js_kernel.h"
namespace onnxruntime {
namespace js {
template <bool is_channels_last>
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
                                   "outputPadding" : $10 ? Array.from(HEAP32.subarray($11, $11 + $10)) : [],
                                   "outputShape" : $12 ? Array.from(HEAP32.subarray($13, $13 + $12)) : []
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
      constexpr size_t pads_vec_size = 4;
      constexpr size_t strides_vec_size = 2;
      constexpr size_t dialations_vec_size = 2;
      constexpr size_t kernel_shape_vec_size = 2;
      // First set default values for pads, strides and dialations
      std::vector<int32_t> local_pads(pads_vec_size, 0);
      std::vector<int32_t> local_strides(strides_vec_size, 0);
      std::vector<int32_t> local_dilations(dialations_vec_size, 0);
      std::vector<int32_t> local_kernel_shape;
      std::vector<int32_t> local_output_shape(conv_transpose_attrs_.output_shape.begin(), conv_transpose_attrs_.output_shape.end());
      std::vector<int32_t> local_output_padding(conv_transpose_attrs_.output_padding.begin(), conv_transpose_attrs_.output_padding.end());
      if (conv_transpose_attrs_.kernel_shape_specified) {
        for (size_t i = 0; i < kernel_shape.size() && i < kernel_shape_vec_size; ++i) {
          local_kernel_shape.push_back(gsl::narrow_cast<int32_t>(kernel_shape[i]));
        }
      } else {
        for (size_t i = 0; i < kernel_shape_vec_size; ++i) {
          local_kernel_shape.push_back(0);
        }
      }
      for (size_t i = 0; i < conv_transpose_attrs_.pads.size() && i < pads_vec_size; ++i) {
        local_pads[i] = gsl::narrow_cast<int32_t>(conv_transpose_attrs_.pads[i]);
      }
      for (size_t i = 0; i < conv_transpose_attrs_.dilations.size() && i < dialations_vec_size; ++i) {
        local_dilations[i] = gsl::narrow_cast<int32_t>(conv_transpose_attrs_.dilations[i]);
      }
      for (size_t i = 0; i < conv_transpose_attrs_.strides.size() && i < strides_vec_size; ++i) {
        local_strides[i] = gsl::narrow_cast<int32_t>(conv_transpose_attrs_.strides[i]);
      }
      LOGS_DEFAULT(VERBOSE) << "output_shape = " << conv_transpose_attrs_.output_shape << std::endl;
      LOGS_DEFAULT(VERBOSE) << "output_padding = " << conv_transpose_attrs_.output_padding << std::endl;
      JSEP_INIT_KERNEL_ATTRIBUTE(ConvTranspose, ({
                                   "format" : $7 ? "NHWC" : "NCHW",
                                   "autoPad" : $1,
                                   "dilations" : Array.from(HEAP32.subarray($2, $2 + /* dialations_vec_size */ 2)),
                                   "group" : $3,
                                   "kernelShape" : Array.from(HEAP32.subarray($4, $4 + /* kernel_shape_vec_size */ 2)),
                                   "pads" : Array.from(HEAP32.subarray($5, $5 + /* pads_vec_size */ 4)),
                                   "strides" : Array.from(HEAP32.subarray($6, $6 + /* strides_vec_size */ 2)),
                                   "wIsConst" : () JS_ARROW(!!HEAP8[$8]),
                                   "outputPadding" : ($9 > 0) ? Array.from(HEAP32.subarray($10, $10 + $9)) : [],
                                   "outputShape" : ($11 > 0) ? Array.from(HEAP32.subarray($12, $12 + $11)) : []
                                 }),
                                 static_cast<int32_t>(conv_transpose_attrs_.auto_pad),
                                 reinterpret_cast<int32_t>(local_dilations.data()) >> 2,
                                 static_cast<int32_t>(conv_transpose_attrs_.group),
                                 reinterpret_cast<int32_t>(local_kernel_shape.data()) >> 2,
                                 reinterpret_cast<int32_t>(local_pads.data()) >> 2,
                                 reinterpret_cast<int32_t>(local_strides.data()) >> 2,
                                 static_cast<int32_t>(channels_last),
                                 reinterpret_cast<int32_t>(&w_is_const_),
                                 gsl::narrow_cast<int32_t>(local_output_padding.size()),
                                 reinterpret_cast<int32_t>(local_output_padding.size() > 0 ? local_output_padding.data() : nullptr) >> 2,
                                 gsl::narrow_cast<int32_t>(local_output_shape.size()),
                                 reinterpret_cast<int32_t>(local_output_shape.size() > 0 ? local_output_shape.data() : nullptr) >> 2);
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
  ConvTransposeAttributes conv_transpose_attrs_;
  bool w_is_const_;
};

}  // namespace js
}  // namespace onnxruntime
