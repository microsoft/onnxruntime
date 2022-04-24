// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/tensor_shape.h"
#include "core/xnnpack/op.h"

namespace onnxruntime {
namespace xnnpack {
class MaxPool2D : public OpKernel {
 public:
  MaxPool2D(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  XNNPackOperator op0_ = nullptr;
  TensorShape output_shape_;
  bool has_const_output_shape_;

  uint32_t input_padding_top_;
  uint32_t input_padding_right_;
  uint32_t input_padding_bottom_;
  uint32_t input_padding_left_;
  uint32_t pooling_height_;
  uint32_t pooling_width_;
  uint32_t stride_height_;
  uint32_t stride_width_;
  uint32_t dilation_height_;
  uint32_t dilation_width_;
  size_t channels_;
  int64_t padding_mode_;
};
}  // namespace xnnpack
}  // namespace onnxruntime