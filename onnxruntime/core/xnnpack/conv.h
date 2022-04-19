// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/xnnpack/op.h"

namespace onnxruntime {
namespace xnnpack {
class Convolution2d : public OpKernel {
 public:
  Convolution2d(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  XNNPackOperator op0_ = nullptr;
  TensorShape output_shape_;
  bool has_const_output_shape_;
  // The following vars are valid only when has_const_output_shape_ == false;
  uint32_t input_padding_top_ = 0;
  uint32_t input_padding_right_ = 0;
  uint32_t input_padding_bottom_ = 0;
  uint32_t input_padding_left_ = 0;
  uint32_t subsampling_height_ = 0;
  uint32_t subsampling_width_ = 0;
  uint32_t dilation_height_ = 0;
  uint32_t dilation_width_ = 0;
  int padding_mode_ = 0;
};

class DepthWiseConvolution2d : public OpKernel {
 public:
  DepthWiseConvolution2d(const OpKernelInfo& info);
  Status Compute(OpKernelContext*) const override;
  ~DepthWiseConvolution2d() { cpu_allocator_->Free(weight_); }

 private:
  XNNPackOperator op0_ = nullptr;
  TensorShape output_shape_;
  bool has_const_output_shape_;
  AllocatorPtr cpu_allocator_;
  // Tranposed weight
  float* weight_ = nullptr;
  // The following vars are valid only when has_const_output_shape_ == false;
  uint32_t input_padding_top_ = 0;
  uint32_t input_padding_right_ = 0;
  uint32_t input_padding_bottom_ = 0;
  uint32_t input_padding_left_ = 0;
  uint32_t subsampling_height_ = 0;
  uint32_t subsampling_width_ = 0;
  uint32_t dilation_height_ = 0;
  uint32_t dilation_width_ = 0;
  int padding_mode_ = 0;
};
}  // namespace xnnpack
}  // namespace onnxruntime
