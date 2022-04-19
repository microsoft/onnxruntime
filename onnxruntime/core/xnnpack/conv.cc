// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/xnnpack/conv.h"

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/xnnpack/build_kernel_info.h"
#include "core/xnnpack/schema/xnnpack_onnx_defs.h"
#include "core/xnnpack/shape_helper.h"

namespace onnxruntime {
namespace xnnpack {

Status XnnPackConvShapeInferKernelImpl(const TensorShape& input_shape, const TensorShape& weight_shape,
                                       uint32_t input_padding_top, uint32_t input_padding_right,
                                       uint32_t input_padding_bottom, uint32_t input_padding_left,
                                       uint32_t subsampling_height, uint32_t subsampling_width, uint32_t dilation_h,
                                       uint32_t dilation_w, ONNXRUNTIME_XNNPACK_PADDING_MODE padding_mode,
                                       std::array<int64_t, 4>& output_dims) {
  if (input_shape.NumDimensions() != 4) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Input tensor must have 4 dimensions");
  }

  if (weight_shape.NumDimensions() != 4) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Weight tensor must have 4 dimensions");
  }

  int64_t input_H_value = input_shape[1];
  int64_t input_W_value = input_shape[2];

  int64_t input_C = input_shape[3];

  int64_t filter_height = weight_shape[1];
  int64_t filter_width = weight_shape[2];
  int64_t in_channels = weight_shape[3];
  input_H_value += static_cast<int64_t>(input_padding_top) + input_padding_bottom;
  input_W_value += static_cast<int64_t>(input_padding_right) + input_padding_left;

  OnnxStatus st = ConvShapeInference(input_shape[0], input_H_value, input_W_value, input_C, weight_shape[0],
                                     filter_height, filter_width, in_channels, subsampling_height, subsampling_width,
                                     dilation_h, dilation_w, padding_mode, output_dims);
  if (!st.IsOK()) {
    return Status(common::ONNXRUNTIME, common::FAIL, st.ErrorMessage());
  }

  return Status::OK();
}

Convolution2d::Convolution2d(const OpKernelInfo& info) : OpKernel(info) {
  const Tensor* weight = nullptr;
  const Tensor* B = nullptr;
  const ONNX_NAMESPACE::TypeProto* input_type_proto = info.GetInputType(0);
  const ONNX_NAMESPACE::TypeProto* output_type_proto = info.GetOutputType(0);

  ORT_ENFORCE(input_type_proto != nullptr && input_type_proto->has_tensor_type() &&
              input_type_proto->tensor_type().has_shape());

  if (output_type_proto != nullptr && output_type_proto->has_tensor_type() &&
      output_type_proto->tensor_type().has_shape()) {
    output_shape_ = utils::GetTensorShapeFromTensorShapeProto(output_type_proto->tensor_type().shape());
    has_const_output_shape_ = IsAllDimKnown(output_shape_);
  } else {
    // It's ok. We will infer it in this->Compute() function.
    has_const_output_shape_ = false;
  }

  ORT_ENFORCE(info.TryGetConstantInput(1, &weight));
  ORT_ENFORCE(info.TryGetConstantInput(2, &B));

  int64_t input_channels = input_type_proto->tensor_type().shape().dim(3).dim_value();
  const TensorShape& kernel_shape = weight->Shape();
  int64_t output_channels = kernel_shape[0];

  int64_t kernel_height = kernel_shape[1];
  int64_t kernel_width = kernel_shape[2];

  int64_t input_padding_top;
  int64_t input_padding_right;
  int64_t input_padding_bottom;
  int64_t input_padding_left;

  int64_t subsampling_height;
  int64_t subsampling_width;
  int64_t dilation_height;
  int64_t dilation_width;
  int64_t groups;
  float output_min;
  float output_max;
  int64_t padding_mode;
  ORT_ENFORCE(info.GetAttr("input_padding_top", &input_padding_top).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_right", &input_padding_right).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_bottom", &input_padding_bottom).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_left", &input_padding_left).IsOK());
  ORT_ENFORCE(info.GetAttr("subsampling_height", &subsampling_height).IsOK());
  ORT_ENFORCE(info.GetAttr("subsampling_width", &subsampling_width).IsOK());
  ORT_ENFORCE(info.GetAttr("dilation_height", &dilation_height).IsOK());
  ORT_ENFORCE(info.GetAttr("dilation_width", &dilation_width).IsOK());
  ORT_ENFORCE(info.GetAttr("groups", &groups).IsOK());
  ORT_ENFORCE(info.GetAttr("output_min", &output_min).IsOK());
  ORT_ENFORCE(info.GetAttr("output_max", &output_max).IsOK());
  ORT_ENFORCE(info.GetAttr("padding_mode", &padding_mode).IsOK());

  input_padding_top_ = gsl::narrow<uint32_t>(input_padding_top);
  input_padding_right_ = gsl::narrow<uint32_t>(input_padding_right);
  input_padding_bottom_ = gsl::narrow<uint32_t>(input_padding_bottom);
  input_padding_left_ = gsl::narrow<uint32_t>(input_padding_left);
  subsampling_height_ = gsl::narrow<uint32_t>(subsampling_height);
  subsampling_width_ = gsl::narrow<uint32_t>(subsampling_width);
  dilation_height_ = gsl::narrow<uint32_t>(dilation_height);
  dilation_width_ = gsl::narrow<uint32_t>(dilation_width);
  padding_mode_ = gsl::narrow<uint32_t>(padding_mode);

  uint32_t flags = 0;
  if (padding_mode == 1) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }
  size_t group_input_channels = input_channels / groups;
  size_t group_output_channels = output_channels / groups;
  xnn_status status;
  struct xnn_operator* p;
  status = xnn_create_convolution2d_nhwc_f32(
      input_padding_top_, input_padding_right_, input_padding_bottom_, input_padding_left_,
      gsl::narrow<uint32_t>(kernel_height), gsl::narrow<uint32_t>(kernel_width), subsampling_height_,
      subsampling_width_, dilation_height_, dilation_width_, gsl::narrow<uint32_t>(groups),
      gsl::narrow<uint32_t>(group_input_channels), gsl::narrow<uint32_t>(group_output_channels),
      gsl::narrow<uint32_t>(input_channels), gsl::narrow<uint32_t>(output_channels), weight->Data<float>(),
      B->Data<float>(), output_min, output_max, flags, &p);
  ORT_ENFORCE(status == xnn_status_success);
  op0_.reset(p);
}

static ONNX_NAMESPACE::TensorShapeProto ToTensorShapeProto(const TensorShape& s) {
  ONNX_NAMESPACE::TensorShapeProto ret;
  size_t len = s.NumDimensions();
  for (size_t i = 0; i != len; ++i) {
    assert(s[i] >= 0);
    ret.add_dim()->set_dim_value(s[i]);
  }
  return ret;
}

Status Convolution2d::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  Tensor* Y = nullptr;
  if (has_const_output_shape_) {
    Y = context->Output(0, output_shape_);
  } else {
    std::array<int64_t, 4> output_dims;
    Status status = XnnPackConvShapeInferKernelImpl(
        X->Shape(), context->Input<Tensor>(1)->Shape(), input_padding_top_, input_padding_right_, input_padding_bottom_,
        input_padding_left_, subsampling_height_, subsampling_width_, dilation_height_, dilation_width_,
        static_cast<ONNXRUNTIME_XNNPACK_PADDING_MODE>(padding_mode_), output_dims);
    if (!status.IsOK()) {
      return Status(common::ONNXRUNTIME, common::FAIL, status.ErrorMessage());
    }
    Y = context->Output(0, TensorShape(output_dims));
  }

  const TensorShape& input_shape = X->Shape();
  xnn_status status = xnn_setup_convolution2d_nhwc_f32(
      op0_.get(), input_shape[0] /* batch size */, input_shape[1] /* input height */, input_shape[2] /* input width */,
      X->Data<float>() /* input */, Y->MutableData<float>() /* output */, nullptr /* threadpool */);
  ORT_ENFORCE(status == xnn_status_success);
  status = xnn_run_operator(op0_.get(), nullptr);
  ORT_ENFORCE(status == xnn_status_success);
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(XnnPackConvolution2d, kMSDomain, 1, kCpuExecutionProvider, KernelDefBuilder(), Convolution2d);

ONNX_OPERATOR_KERNEL_EX(XnnPackDepthwiseConvolution2d, kMSDomain, 1, kCpuExecutionProvider, KernelDefBuilder(),
                        DepthWiseConvolution2d);

Status DepthWiseConvolution2d::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  Tensor* Y = nullptr;
  if (has_const_output_shape_) {
    Y = context->Output(0, output_shape_);
  } else {
    const ONNX_NAMESPACE::TensorShapeProto* weight_shape = Node().InputDefs()[1]->Shape();
    ORT_ENFORCE(weight_shape != nullptr);
    const ONNX_NAMESPACE::TensorShapeProto input_shape = ToTensorShapeProto(X->Shape());
    ONNX_NAMESPACE::TensorShapeProto final_output_shape;

    OnnxStatus status = XnnPackDepthwiseConvolution2dShapeInferImpl(
        input_shape, *weight_shape, input_padding_top_, input_padding_right_, input_padding_bottom_,
        input_padding_left_, subsampling_height_, subsampling_width_, dilation_height_, dilation_width_,
        static_cast<ONNXRUNTIME_XNNPACK_PADDING_MODE>(padding_mode_), &final_output_shape);
    if (!status.IsOK()) {
      return Status(common::ONNXRUNTIME, common::FAIL, status.ErrorMessage());
    }
    TensorShape output_shape = utils::GetTensorShapeFromTensorShapeProto(final_output_shape);
    if (!IsAllDimKnown(output_shape)) {
      // If it happens, we have a logic error
      return Status(common::ONNXRUNTIME, common::FAIL, "Cannot infer output shape");
    }
    Y = context->Output(0, output_shape);
  }
  const TensorShape& input_shape = X->Shape();
  xnn_status status = xnn_setup_convolution2d_nhwc_f32(
      op0_.get(), input_shape[0] /* batch size */, input_shape[1] /* input height */, input_shape[2] /* input width */,
      X->Data<float>() /* input */, Y->MutableData<float>() /* output */, nullptr /* threadpool */);
  ORT_ENFORCE(status == xnn_status_success);
  status = xnn_run_operator(op0_.get(), nullptr);
  ORT_ENFORCE(status == xnn_status_success);

  return Status::OK();
}

static void hwc_to_chw(const float* input, size_t h, size_t w, size_t channels, float* output_data) {
  size_t stride = h * w;
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != channels; ++c) {
      output_data[c * stride + i] = input[i * channels + c];
    }
  }
}
DepthWiseConvolution2d::DepthWiseConvolution2d(const OpKernelInfo& info) : OpKernel(info) {
  const Tensor* weight = nullptr;
  const Tensor* B = nullptr;
  const ONNX_NAMESPACE::TypeProto* input_type_proto = info.GetInputType(0);
  const ONNX_NAMESPACE::TypeProto* output_type_proto = info.GetOutputType(0);
  ORT_ENFORCE(input_type_proto != nullptr);
  ORT_ENFORCE(output_type_proto != nullptr);
  output_shape_ = utils::GetTensorShapeFromTensorShapeProto(output_type_proto->tensor_type().shape());
  has_const_output_shape_ = IsAllDimKnown(output_shape_);

  ORT_ENFORCE(info.TryGetConstantInput(1, &weight));
  ORT_ENFORCE(info.TryGetConstantInput(2, &B));
  const TensorShape& kernel_shape = weight->Shape();
  cpu_allocator_ = info.GetAllocator(0, OrtMemTypeDefault);
  weight_ = static_cast<float*>(cpu_allocator_->AllocArray(kernel_shape.Size(), sizeof(float)));
  ORT_ENFORCE(weight_ != nullptr);
  TensorShape new_weight_shape{kernel_shape[3], kernel_shape[1], kernel_shape[2], 1};
  hwc_to_chw(weight->Data<float>(), kernel_shape[1], kernel_shape[2], kernel_shape[3], weight_);

  int64_t input_channels = input_type_proto->tensor_type().shape().dim(3).dim_value();
  // Weight shape : [ 1, kernel_height, kernel_width, input_channels * depth_multiplier ]
  ORT_ENFORCE(kernel_shape.NumDimensions() == 4);
  ORT_ENFORCE(kernel_shape[3] % input_channels == 0);
  int64_t kernel_height = kernel_shape[1];
  int64_t kernel_width = kernel_shape[2];

  int64_t input_padding_top;
  int64_t input_padding_right;
  int64_t input_padding_bottom;
  int64_t input_padding_left;

  int64_t subsampling_height;
  int64_t subsampling_width;
  int64_t dilation_height;
  int64_t dilation_width;
  float output_min;
  float output_max;
  int64_t padding_mode;
  ORT_ENFORCE(info.GetAttr("input_padding_top", &input_padding_top).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_right", &input_padding_right).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_bottom", &input_padding_bottom).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_left", &input_padding_left).IsOK());
  ORT_ENFORCE(info.GetAttr("subsampling_height", &subsampling_height).IsOK());
  ORT_ENFORCE(info.GetAttr("subsampling_width", &subsampling_width).IsOK());
  ORT_ENFORCE(info.GetAttr("dilation_height", &dilation_height).IsOK());
  ORT_ENFORCE(info.GetAttr("dilation_width", &dilation_width).IsOK());
  // TODO: handle optional case
  ORT_ENFORCE(info.GetAttr("output_min", &output_min).IsOK());
  ORT_ENFORCE(info.GetAttr("output_max", &output_max).IsOK());
  ORT_ENFORCE(info.GetAttr("padding_mode", &padding_mode).IsOK());

  input_padding_top_ = gsl::narrow<uint32_t>(input_padding_top);
  input_padding_right_ = gsl::narrow<uint32_t>(input_padding_right);
  input_padding_bottom_ = gsl::narrow<uint32_t>(input_padding_bottom);
  input_padding_left_ = gsl::narrow<uint32_t>(input_padding_left);
  subsampling_height_ = gsl::narrow<uint32_t>(subsampling_height);
  subsampling_width_ = gsl::narrow<uint32_t>(subsampling_width);
  dilation_height_ = gsl::narrow<uint32_t>(dilation_height);
  dilation_width_ = gsl::narrow<uint32_t>(dilation_width);
  padding_mode_ = gsl::narrow<uint32_t>(padding_mode);

  uint32_t flags = 0;
  if (padding_mode == 1) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }
  int64_t depth_multiplier = kernel_shape[3] / input_channels;
  struct xnn_operator* p;
  xnn_status status = xnn_create_convolution2d_nhwc_f32(
      input_padding_top_, input_padding_right_, input_padding_bottom_, input_padding_left_,
      gsl::narrow<uint32_t>(kernel_height), gsl::narrow<uint32_t>(kernel_width), subsampling_height_,
      subsampling_width_, dilation_height_, dilation_width_, gsl::narrow<uint32_t>(input_channels) /* groups */,
      1 /* group_input_channels */, depth_multiplier /* group_output_channels */,
      input_channels /* input_channel_stride */, kernel_shape[3] /* output_channel_stride */, weight_, B->Data<float>(),
      output_min, output_max, flags, &p);
  ORT_ENFORCE(status == xnn_status_success);
  op0_.reset(p);
}

}  // namespace xnnpack
}  // namespace onnxruntime
