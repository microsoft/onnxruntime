// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/xnnpack/max_pooling_2d.h"

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/xnnpack/build_kernel_info.h"
#include "core/xnnpack/schema/xnnpack_onnx_defs.h"
#include "core/xnnpack/shape_helper.h"

namespace onnxruntime {
namespace xnnpack {

MaxPool2D::MaxPool2D(const OpKernelInfo& info) : OpKernel(info) {
  const ONNX_NAMESPACE::TypeProto* input_type_proto = info.GetInputType(0);
  const ONNX_NAMESPACE::TypeProto* output_type_proto = info.GetOutputType(0);

  ORT_ENFORCE(input_type_proto != nullptr && input_type_proto->has_tensor_type() &&
              input_type_proto->tensor_type().has_shape());
  ORT_ENFORCE(output_type_proto != nullptr && output_type_proto->has_tensor_type());

  if (output_type_proto->tensor_type().has_shape()) {
    output_shape_ = utils::GetTensorShapeFromTensorShapeProto(output_type_proto->tensor_type().shape());
    has_const_output_shape_ = IsAllDimKnown(output_shape_);
  } else {
    has_const_output_shape_ = false;
  }

  int64_t input_padding_top;
  int64_t input_padding_right;
  int64_t input_padding_bottom;
  int64_t input_padding_left;

  int64_t pooling_height;
  int64_t pooling_width;
  int64_t stride_height;
  int64_t stride_width;
  int64_t dilation_height;
  int64_t dilation_width;

  float output_min;
  float output_max;
  ORT_ENFORCE(info.GetAttr("input_padding_top", &input_padding_top).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_right", &input_padding_right).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_bottom", &input_padding_bottom).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_left", &input_padding_left).IsOK());
  ORT_ENFORCE(info.GetAttr("pooling_height", &pooling_height).IsOK());
  ORT_ENFORCE(info.GetAttr("pooling_width", &pooling_width).IsOK());
  ORT_ENFORCE(info.GetAttr("stride_height", &stride_height).IsOK());
  ORT_ENFORCE(info.GetAttr("stride_width", &stride_width).IsOK());
  ORT_ENFORCE(info.GetAttr("dilation_height", &dilation_height).IsOK());
  ORT_ENFORCE(info.GetAttr("dilation_width", &dilation_width).IsOK());

  ORT_ENFORCE(info.GetAttr("output_min", &output_min).IsOK());
  ORT_ENFORCE(info.GetAttr("output_max", &output_max).IsOK());
  ORT_ENFORCE(info.GetAttr("padding_mode", &padding_mode_).IsOK());

  input_padding_top_ = gsl::narrow<uint32_t>(input_padding_top);
  input_padding_right_ = gsl::narrow<uint32_t>(input_padding_right);
  input_padding_bottom_ = gsl::narrow<uint32_t>(input_padding_bottom);
  input_padding_left_ = gsl::narrow<uint32_t>(input_padding_left);
  pooling_height_ = gsl::narrow<uint32_t>(pooling_height);
  pooling_width_ = gsl::narrow<uint32_t>(pooling_width);
  stride_height_ = gsl::narrow<uint32_t>(stride_height);
  stride_width_ = gsl::narrow<uint32_t>(stride_width);
  dilation_height_ = gsl::narrow<uint32_t>(dilation_height);
  dilation_width_ = gsl::narrow<uint32_t>(dilation_width);
  uint32_t flags = 0;
  if (padding_mode_ == 1) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }
  int64_t input_channels = input_type_proto->tensor_type().shape().dim(3).dim_value();

  xnn_status status;
  struct xnn_operator* p;
  status = xnn_create_max_pooling2d_nhwc_f32(input_padding_top_, input_padding_right_, input_padding_bottom_,
                                             input_padding_left_, pooling_height_, pooling_width_, stride_height_,
                                             stride_width_, dilation_height_, dilation_width_,
                                             static_cast<size_t>(input_channels), static_cast<size_t>(input_channels),
                                             static_cast<size_t>(input_channels), output_min, output_max, flags, &p);
  ORT_ENFORCE(status == xnn_status_success);
  op0_.reset(p);
}

Status MaxPool2D::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  Tensor* Y = nullptr;
  if (has_const_output_shape_) {
    Y = context->Output(0, output_shape_);
  } else {
    std::array<int64_t, 4> output_dims;
    const auto& input_shape = X->Shape();
    if (input_shape.NumDimensions() != 4) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Input tensor must have 4 dimensions");
    }

    int64_t input_H_value = input_shape[1];
    int64_t input_W_value = input_shape[2];

    input_H_value += static_cast<int64_t>(input_padding_top_) + input_padding_bottom_;
    input_W_value += static_cast<int64_t>(input_padding_right_) + input_padding_left_;

    OnnxStatus status =
        MaxPoolShapeInference(input_shape[0], input_H_value, input_W_value, input_shape[3], pooling_height_,
                              pooling_width_, stride_height_, stride_width_, dilation_height_, dilation_width_,
                              static_cast<ONNXRUNTIME_XNNPACK_PADDING_MODE>(padding_mode_), output_dims);
    if (!status.IsOK()) {
      return Status(common::ONNXRUNTIME, common::FAIL, status.ErrorMessage());
    }
    Y = context->Output(0, TensorShape(output_dims));
  }

  const TensorShape& input_shape = X->Shape();
  xnn_status status = xnn_setup_max_pooling2d_nhwc_f32(
      op0_.get(), input_shape[0] /* batch size */, input_shape[1] /* input height */, input_shape[2] /* input width */,
      X->Data<float>() /* input */, Y->MutableData<float>() /* output */, nullptr /* threadpool */);
  ORT_ENFORCE(status == xnn_status_success);
  status = xnn_run_operator(op0_.get(), nullptr);
  ORT_ENFORCE(status == xnn_status_success);
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(XnnPackMaxPooling2d, kMSDomain, 1, kCpuExecutionProvider, KernelDefBuilder(), MaxPool2D);

}  // namespace xnnpack
}  // namespace onnxruntime