// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xnnpack_onnx_defs.h"

#include <onnx/defs/schema.h>

#include <array>
#include <safeint/SafeInt.hpp>

using namespace onnx;

namespace onnxruntime {
namespace xnnpack {

using ::ONNX_NAMESPACE::Common::StatusCategory;
using ::ONNX_NAMESPACE::Common::StatusCode;

OnnxStatus ComputeOutputSizeSame(ptrdiff_t input_size, uint32_t stride, ptrdiff_t* output_size) {
  if (stride == 0 || input_size <= 0) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }
  size_t r = static_cast<size_t>(input_size) + stride - 1;
  if (r > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }
  *output_size = r / stride;
  return OnnxStatus::OK();
}

OnnxStatus ComputeOutputSizeValid(ptrdiff_t input_size, uint32_t stride, ptrdiff_t filter_size, uint32_t dilation_rate,
                                  ptrdiff_t* output_size) {
  if (filter_size < 1) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }
  if (stride <= 0) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }
  if (dilation_rate > 1) {
    if (!SafeMultiply(filter_size - 1, dilation_rate, filter_size)) {
      return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
    }
    if (!SafeAdd(filter_size, 1, filter_size)) {
      return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
    }
  }

  if (!SafeSubtract(input_size, filter_size, input_size)) return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  if (input_size < 0) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }
  if (!SafeAdd(input_size, stride, input_size)) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }
  assert(input_size > 0);
  if (!SafeDivide(input_size, stride, input_size)) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }
  *output_size = input_size;
  return ::ONNX_NAMESPACE::Common::Status::OK();
}

OnnxStatus XnnPackConvShapeInferImpl(const ::ONNX_NAMESPACE::TensorShapeProto& input_shape,
                                     const ::ONNX_NAMESPACE::TensorShapeProto& weight_shape, uint32_t input_padding_top,
                                     uint32_t input_padding_right, uint32_t input_padding_bottom,
                                     uint32_t input_padding_left, uint32_t subsampling_height,
                                     uint32_t subsampling_width, uint32_t dilation_h, uint32_t dilation_w,
                                     ONNXRUNTIME_XNNPACK_PADDING_MODE padding_mode,
                                     ::ONNX_NAMESPACE::TensorShapeProto* final_output_shape) {
  if (input_shape.dim_size() != 4) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Input tensor must have 4 dimensions");
  }

  if (weight_shape.dim_size() != 4) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Weight tensor must have 4 dimensions");
  }

  if (!input_shape.dim(3).has_dim_value()) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Channel can not be unknown");
  }
  // Though the first dim of weight can be unknown, our implementation doesn't support it.
  for (int i = 1; i != 3; ++i) {
    if (!weight_shape.dim(i).has_dim_value()) {
      return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Only the first dim can be unknown");
    }
  }
  const ::ONNX_NAMESPACE::TensorShapeProto_Dimension& input_H = input_shape.dim(1);
  const ::ONNX_NAMESPACE::TensorShapeProto_Dimension& input_W = input_shape.dim(2);
  std::array<::ONNX_NAMESPACE::TensorShapeProto_Dimension*, 4> output_dims = {nullptr, nullptr, nullptr, nullptr};

  output_dims[0] = final_output_shape->add_dim();
  output_dims[1] = final_output_shape->add_dim();
  output_dims[2] = final_output_shape->add_dim();
  output_dims[3] = final_output_shape->add_dim();
  if (input_H.has_dim_value() && input_W.has_dim_value()) {
    int64_t input_C = input_shape.dim(3).dim_value();

    int64_t filter_height = weight_shape.dim(1).dim_value();
    int64_t filter_width = weight_shape.dim(2).dim_value();
    int64_t in_channels = weight_shape.dim(3).dim_value();
    int64_t input_H_value = input_H.dim_value();
    int64_t input_W_value = input_W.dim_value();
    input_H_value += static_cast<int64_t>(input_padding_top) + input_padding_bottom;
    input_W_value += static_cast<int64_t>(input_padding_right) + input_padding_left;

    ONNX_RETURN_IF_ERROR(ConvShapeInference(
        input_shape.dim(0), input_H_value, input_W_value, input_C, weight_shape.dim(0), filter_height, filter_width,
        in_channels, subsampling_height, subsampling_width, dilation_h, dilation_w, padding_mode, output_dims));
  } else {
    *output_dims[0] = input_shape.dim(0);
    *output_dims[3] = weight_shape.dim(0);
  }

  return OnnxStatus::OK();
}

OnnxStatus XnnPackMaxPoolShapeInferImpl(const ::ONNX_NAMESPACE::TensorShapeProto& input_shape, int64_t pooling_height,
                                        int64_t pooling_width, uint32_t input_padding_top, uint32_t input_padding_right,
                                        uint32_t input_padding_bottom, uint32_t input_padding_left, uint32_t strides_h,
                                        uint32_t strides_w, uint32_t dilation_h, uint32_t dilation_w,
                                        ONNXRUNTIME_XNNPACK_PADDING_MODE padding_mode,
                                        ::ONNX_NAMESPACE::TensorShapeProto* final_output_shape) {
  if (input_shape.dim_size() != 4) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Input tensor must have 4 dimensions");
  }

  if (!input_shape.dim(3).has_dim_value()) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Channel can not be unknown");
  }

  const ::ONNX_NAMESPACE::TensorShapeProto_Dimension& input_H = input_shape.dim(1);
  const ::ONNX_NAMESPACE::TensorShapeProto_Dimension& input_W = input_shape.dim(2);
  std::array<::ONNX_NAMESPACE::TensorShapeProto_Dimension*, 4> output_dims = {nullptr, nullptr, nullptr, nullptr};

  output_dims[0] = final_output_shape->add_dim();
  output_dims[1] = final_output_shape->add_dim();
  output_dims[2] = final_output_shape->add_dim();
  output_dims[3] = final_output_shape->add_dim();
  if (input_H.has_dim_value() && input_W.has_dim_value()) {
    int64_t input_C = input_shape.dim(3).dim_value();
    int64_t input_H_value = input_H.dim_value();
    int64_t input_W_value = input_W.dim_value();
    input_H_value += static_cast<int64_t>(input_padding_top) + input_padding_bottom;
    input_W_value += static_cast<int64_t>(input_padding_right) + input_padding_left;

    MaxPoolShapeInference(input_shape.dim(0), input_H_value, input_W_value, input_C, pooling_height, pooling_width,
                          strides_h, strides_w, dilation_h, dilation_w, padding_mode, output_dims);
  } else {
    // N stays the same.
    *output_dims[0] = input_shape.dim(0);
    // Channel stays the same. The following line is necessary, otherwise if this node is followed by a Conv then the
    // input channel dim of conv could become unknown after ONNX -> XNNPack conversion.
    *output_dims[3] = input_shape.dim(3);
  }
  return OnnxStatus::OK();
}

OnnxStatus XnnPackDepthwiseConvolution2dShapeInferImpl(const ::ONNX_NAMESPACE::TensorShapeProto& input_shape,
                                                       const ::ONNX_NAMESPACE::TensorShapeProto& weight_shape,
                                                       uint32_t input_padding_top, uint32_t input_padding_right,
                                                       uint32_t input_padding_bottom, uint32_t input_padding_left,
                                                       uint32_t subsampling_height, uint32_t subsampling_width,
                                                       uint32_t dilation_h, uint32_t dilation_w,
                                                       ONNXRUNTIME_XNNPACK_PADDING_MODE padding_mode,
                                                       ::ONNX_NAMESPACE::TensorShapeProto* final_output_shape) {
  if (input_shape.dim_size() != 4) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Input tensor must have 4 dimensions");
  }

  if (weight_shape.dim_size() != 4) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Weight tensor must have 4 dimensions");
  }

  const ::ONNX_NAMESPACE::TensorShapeProto_Dimension& input_H = input_shape.dim(1);
  const ::ONNX_NAMESPACE::TensorShapeProto_Dimension& input_W = input_shape.dim(2);
  int64_t input_C = input_shape.dim(3).dim_value();

  if (input_C == 0) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Input channel can not be zero");
  }

  // Weight shape: [1, kernel_height, kernel_width, input_channels * depth_multiplier]
  int64_t size_one = weight_shape.dim(0).dim_value();
  if (size_one != 1) {
    std::ostringstream oss;
    oss << "The first dim of weight must be one. Got " << size_one << ".";
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, oss.str());
  }
  int64_t filter_height = weight_shape.dim(1).dim_value();
  int64_t filter_width = weight_shape.dim(2).dim_value();
  if (weight_shape.dim(3).has_dim_value()) {
    int64_t input_channels_by_depth_multiplier = weight_shape.dim(3).dim_value();
    if (input_channels_by_depth_multiplier % input_C != 0) {
      return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL,
                        "The last dim of weight is not multiple of input channels.");
    }
  }
  std::array<::ONNX_NAMESPACE::TensorShapeProto_Dimension*, 4> output_dims = {nullptr, nullptr, nullptr, nullptr};

  output_dims[0] = final_output_shape->add_dim();
  output_dims[1] = final_output_shape->add_dim();
  output_dims[2] = final_output_shape->add_dim();
  output_dims[3] = final_output_shape->add_dim();

  if (input_H.has_dim_value() && input_W.has_dim_value()) {
    int64_t input_H_value = input_H.dim_value();
    int64_t input_W_value = input_W.dim_value();
    input_H_value += static_cast<int64_t>(input_padding_top) + input_padding_bottom;
    input_W_value += static_cast<int64_t>(input_padding_right) + input_padding_left;
    ONNX_RETURN_IF_ERROR(ConvShapeInference(
        input_shape.dim(0), input_H_value, input_W_value, input_C, weight_shape.dim(3), filter_height, filter_width,
        input_C, subsampling_height, subsampling_width, dilation_h, dilation_w, padding_mode, output_dims));
  } else {
    *output_dims[0] = input_shape.dim(0);
    *output_dims[3] = weight_shape.dim(3);
  }

  return OnnxStatus::OK();
}

// Compare to the signatures of xnn_define_convolution_2d function, this schema doesn't have
// 1. kernel_height. Because it is just a dimension size of the weights
// 2. kernel_width. Because it is just a dimension size of the weights
// 3. group_input_channels. number of input channels per group. Can be calculated if input channels and the number of
// groups are known
// 4. group_output_channels. As the above
ONNX_XNNPACK_OPERATOR_SET_SCHEMA(
    XnnPackConvolution2d, 1,
    OpSchema()
        .Input(0, "X", "", "tensor(float)")
        .Input(1, "W", "", "tensor(float)")
        .Input(2, "B", "", "tensor(float)")
        .Output(0, "Y", "", "tensor(float)")
        .Attr("input_padding_top", "Implicit zero-padding above 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_right",
              "Implicit zero-padding to the right of 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_bottom", "Implicit zero-padding below 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_left",
              "Implicit zero-padding to the left of 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("subsampling_height", "subsampling_height. TFLite stride_height", AttributeProto::INT)
        .Attr("subsampling_width", "subsampling_width. TFLite stride_width", AttributeProto::INT)
        .Attr("dilation_height", "dilation_height. TFLite dilation_height_factor", AttributeProto::INT)
        .Attr("dilation_width", "dilation_width. TFLite dilation_width_factor", AttributeProto::INT)
        .Attr("groups", "groups", AttributeProto::INT)
        .Attr("padding_mode", "0:VALID. 1:SAME.", AttributeProto::INT)
        .Attr("output_min", "output_min", AttributeProto::FLOAT, -INFINITY)
        .Attr("output_max", "output_max", AttributeProto::FLOAT, INFINITY)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto input_shape = ctx.getInputType(0)->tensor_type().shape();
          auto weight_shape = ctx.getInputType(1)->tensor_type().shape();
          uint32_t input_padding_top = static_cast<uint32_t>(getAttribute(ctx, "input_padding_top", 0));
          uint32_t input_padding_right = static_cast<uint32_t>(getAttribute(ctx, "input_padding_right", 0));
          uint32_t input_padding_bottom = static_cast<uint32_t>(getAttribute(ctx, "input_padding_bottom", 0));
          uint32_t input_padding_left = static_cast<uint32_t>(getAttribute(ctx, "input_padding_left", 0));

          uint32_t subsampling_height = static_cast<uint32_t>(getAttribute(ctx, "subsampling_height", 0));
          uint32_t subsampling_width = static_cast<uint32_t>(getAttribute(ctx, "subsampling_width", 0));
          uint32_t dilation_height = static_cast<uint32_t>(getAttribute(ctx, "dilation_height", 0));
          uint32_t dilation_width = static_cast<uint32_t>(getAttribute(ctx, "dilation_width", 0));
          ONNXRUNTIME_XNNPACK_PADDING_MODE padding_mode =
              static_cast<ONNXRUNTIME_XNNPACK_PADDING_MODE>(getAttribute(ctx, "padding_mode", 0));

          auto final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          OnnxStatus status =
              XnnPackConvShapeInferImpl(input_shape, weight_shape, input_padding_top, input_padding_right,
                                        input_padding_bottom, input_padding_left, subsampling_height, subsampling_width,
                                        dilation_height, dilation_width, padding_mode, final_output_shape);
          if (!status.IsOK()) {
            // Convert the status to an exception
            fail_shape_inference(status.ErrorMessage());
          }
        }));

ONNX_XNNPACK_OPERATOR_SET_SCHEMA(
    XnnPackMaxPooling2d, 1,
    OpSchema()
        .Input(0, "X", "", "T")
        .Output(0, "Y", "", "T")
        .TypeConstraint("T",
                        {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int8)", "tensor(int16)",
                         "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)",
                         "tensor(uint64)", "tensor(bool)", "tensor(string)", "tensor(bfloat16)"},
                        "")
        .Attr("input_padding_top", "Implicit zero-padding above 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_right",
              "Implicit zero-padding to the right of 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_bottom", "Implicit zero-padding below 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_left",
              "Implicit zero-padding to the left of 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("pooling_height", "Pooling (kernel) height.", AttributeProto::INT)
        .Attr("pooling_width", "Pooling (kernel) width.", AttributeProto::INT)
        .Attr("stride_height",
              "Displacing of the pooling window in the vertical dimension of the input pixels corresponding to "
              "vertically adjacent output pixels.",
              AttributeProto::INT)
        .Attr("stride_width",
              "Displacing of the pooling window in the horizontal dimension of the input pixels corresponding to "
              "horizontally adjacent output pixels.",
              AttributeProto::INT)
        .Attr("dilation_height", "Dilation of pooling elements along the height dimension.", AttributeProto::INT)
        .Attr("dilation_width", "Dilation of pooling elements along the width dimension.", AttributeProto::INT)
        .Attr("padding_mode", "0:VALID. 1:SAME.", AttributeProto::INT)
        .Attr("output_min", "output_min", AttributeProto::FLOAT, -INFINITY)
        .Attr("output_max", "output_max", AttributeProto::FLOAT, INFINITY)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto input_shape = ctx.getInputType(0)->tensor_type().shape();
          uint32_t input_padding_top = static_cast<uint32_t>(getAttribute(ctx, "input_padding_top", 0));
          uint32_t input_padding_right = static_cast<uint32_t>(getAttribute(ctx, "input_padding_right", 0));
          uint32_t input_padding_bottom = static_cast<uint32_t>(getAttribute(ctx, "input_padding_bottom", 0));
          uint32_t input_padding_left = static_cast<uint32_t>(getAttribute(ctx, "input_padding_left", 0));
          uint32_t pooling_height = static_cast<uint32_t>(getAttribute(ctx, "pooling_height", 0));
          uint32_t pooling_width = static_cast<uint32_t>(getAttribute(ctx, "pooling_width", 0));

          uint32_t stride_height = static_cast<uint32_t>(getAttribute(ctx, "stride_height", 0));
          uint32_t stride_width = static_cast<uint32_t>(getAttribute(ctx, "stride_width", 0));
          uint32_t dilation_height = static_cast<uint32_t>(getAttribute(ctx, "dilation_height", 0));
          uint32_t dilation_width = static_cast<uint32_t>(getAttribute(ctx, "dilation_width", 0));
          ONNXRUNTIME_XNNPACK_PADDING_MODE padding_mode =
              static_cast<ONNXRUNTIME_XNNPACK_PADDING_MODE>(getAttribute(ctx, "padding_mode", 0));

          auto final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          OnnxStatus status = XnnPackMaxPoolShapeInferImpl(
              input_shape, pooling_height, pooling_width, input_padding_top, input_padding_right, input_padding_bottom,
              input_padding_left, stride_height, stride_width, dilation_height, dilation_width, padding_mode,
              final_output_shape);
          if (!status.IsOK()) {
            // Convert the status to an exception
            fail_shape_inference(status.ErrorMessage());
          }
        }));

// Compare to the signatures of xnn_define_convolution_2d function, this schema doesn't have
// 1. kernel_height. Because it is just a dimension size of the weights
// 2. kernel_width. Because it is just a dimension size of the weights
// 3. group_input_channels. number of input channels per group. Can be calculated if input channels and the number of
// groups are known
// 4. group_output_channels. As the above
// 5. depth_multiplier
// Please note this operator uses a different weight layout compared to the normal Convolution2d.
ONNX_XNNPACK_OPERATOR_SET_SCHEMA(
    XnnPackDepthwiseConvolution2d, 1,
    OpSchema()
        .Input(0, "X", "", "tensor(float)")
        .Input(1, "W", "Shape:[1, kernel_height, kernel_width, input_channels * depth_multiplier]", "tensor(float)")
        .Input(2, "B", "", "tensor(float)")
        .Output(0, "Y", "", "tensor(float)")
        .Attr("input_padding_top", "input_padding_top", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_right", "input_padding_right", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_bottom", "input_padding_bottom", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_left", "input_padding_left", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("subsampling_height", "subsampling_height. TFLite stride_height", AttributeProto::INT)
        .Attr("subsampling_width", "subsampling_width. TFLite stride_width", AttributeProto::INT)
        .Attr("dilation_height", "dilation_height. TFLite dilation_height_factor", AttributeProto::INT)
        .Attr("dilation_width", "dilation_width. TFLite dilation_width_factor", AttributeProto::INT)
        .Attr("padding_mode", "0:VALID. 1:SAME.", AttributeProto::INT)
        .Attr("output_min", "output_min", AttributeProto::FLOAT, -INFINITY)
        .Attr("output_max", "output_max", AttributeProto::FLOAT, INFINITY)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto input_shape = ctx.getInputType(0)->tensor_type().shape();
          auto weight_shape = ctx.getInputType(1)->tensor_type().shape();
          uint32_t input_padding_top = static_cast<uint32_t>(getAttribute(ctx, "input_padding_top", 0));
          uint32_t input_padding_right = static_cast<uint32_t>(getAttribute(ctx, "input_padding_right", 0));
          uint32_t input_padding_bottom = static_cast<uint32_t>(getAttribute(ctx, "input_padding_bottom", 0));
          uint32_t input_padding_left = static_cast<uint32_t>(getAttribute(ctx, "input_padding_left", 0));

          uint32_t subsampling_height = static_cast<uint32_t>(getAttribute(ctx, "subsampling_height", 0));
          uint32_t subsampling_width = static_cast<uint32_t>(getAttribute(ctx, "subsampling_width", 0));
          uint32_t dilation_height = static_cast<uint32_t>(getAttribute(ctx, "dilation_height", 0));
          uint32_t dilation_width = static_cast<uint32_t>(getAttribute(ctx, "dilation_width", 0));
          ONNXRUNTIME_XNNPACK_PADDING_MODE padding_mode =
              static_cast<ONNXRUNTIME_XNNPACK_PADDING_MODE>(getAttribute(ctx, "padding_mode", 0));

          auto final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          OnnxStatus status = XnnPackDepthwiseConvolution2dShapeInferImpl(
              input_shape, weight_shape, input_padding_top, input_padding_right, input_padding_bottom,
              input_padding_left, subsampling_height, subsampling_width, dilation_height, dilation_width, padding_mode,
              final_output_shape);
          if (!status.IsOK()) {
            // Convert the status to an exception
            fail_shape_inference(status.ErrorMessage());
          }
        }));
}  // namespace xnnpack
}  // namespace onnxruntime
