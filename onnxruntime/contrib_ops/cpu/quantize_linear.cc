// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantize_linear.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    uint8_t,
    KernelDefBuilder()
        .TypeConstraint("axis", DataTypeImpl::GetType<int64_t>())
        .TypeConstraint("x", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("x_scale", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("x_zero_point", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("y", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<uint8_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    int8_t,
    KernelDefBuilder()
        .TypeConstraint("axis", DataTypeImpl::GetType<int64_t>())
        .TypeConstraint("x", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("x_scale", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("x_zero_point", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("y", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<int8_t>);

template <typename T>
// formula is Y = (X - ZeroPoint) * Scale
Status DequantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& x_scale = *ctx->Input<Tensor>(1);
  auto& x_zero_point = *ctx->Input<Tensor>(2);
  auto& y = *ctx->Output(0, x.Shape());

  const auto& x_shape = x.Shape();
  const auto& scale_shape = x_scale.Shape();
  const auto& zero_point_shape = x_zero_point.Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());

  size_t stride = 0;
  const auto& broadcastDim = x_shape[axis];

  if (has_axis_) {
    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(scale_shape.NumDimensions() == 1 && scale_shape.Size() == broadcastDim, "x_scale must be 1D tensor with size ", broadcastDim);
    ORT_ENFORCE(zero_point_shape.NumDimensions() == 1 && zero_point_shape.Size() == broadcastDim, "x_zero_point must be 1D tensor with size ", broadcastDim);
    stride = 1;
  } else {
    // if no axis, enforce that scale and zero point are scalars
    ORT_ENFORCE(scale_shape.NumDimensions() == 0, "x_scale must be a scalar if no axis is provided");
    ORT_ENFORCE(zero_point_shape.NumDimensions() == 0, "x_zero_point must be a scalar if no axis is provided");
  }

  size_t N = x_shape.SizeToDimension(axis);
  const T* zero_point = x_zero_point.template Data<T>();
  const float* scale = x_scale.template Data<float>();
  size_t block_size = x_shape.SizeFromDimension(axis + 1);
  const T* input = x.template Data<T>();
  float* output = y.template MutableData<float>();

  for (size_t n = 0; n < N; n++) {
    const float* current_scale = scale;
    const T* current_zero_point = zero_point;

    for (size_t bd = 0; bd < static_cast<size_t>(broadcastDim); bd++) {
      auto zp = static_cast<const int>(*current_zero_point);
      auto sc = *current_scale;

      for (size_t bs = 0; bs < block_size; bs++) {
        *output++ = static_cast<float>(static_cast<const int>(*input++) - zp) * sc;
      }

      current_scale += stride;
      current_zero_point += stride;
    }
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("axis", DataTypeImpl::GetType<int64_t>())
        .TypeConstraint("x", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("y_scale", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("y_zero_point", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("y", DataTypeImpl::GetTensorType<uint8_t>()),
    QuantizeLinear<float>);

// clamp doesn't exist in the version of <algorithm> that we're using, so
// make a local one.
static float clamp(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

template <>
// formula is Y = X / Scale + ZeroPoint
Status QuantizeLinear<float>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto& y_zero_point = *ctx->Input<Tensor>(2);
  auto& y = *ctx->Output(0, x.Shape());

  const auto& x_shape = x.Shape();
  const auto& scale_shape = y_scale.Shape();
  const auto& zero_point_shape = y_zero_point.Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());

  size_t stride = 0;
  const auto& broadcastDim = x_shape[axis];

  if (has_axis_) {
    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(scale_shape.NumDimensions() == 1 && scale_shape.Size() == broadcastDim, "x_scale must be 1D tensor with size ", broadcastDim);
    ORT_ENFORCE(zero_point_shape.NumDimensions() == 1 && zero_point_shape.Size() == broadcastDim, "x_zero_point must be 1D tensor with size ", broadcastDim);
    stride = 1;
  } else {
    // if no axis, enforce that scale and zero point are scalars
    ORT_ENFORCE(scale_shape.NumDimensions() == 0, "x_scale must be a scalar if no axis is provided");
    ORT_ENFORCE(zero_point_shape.NumDimensions() == 0, "x_zero_point must be a scalar if no axis is provided");
  }

  size_t N = x_shape.SizeToDimension(axis);
  const uint8_t* zero_point = y_zero_point.template Data<uint8_t>();
  const float* scale = y_scale.template Data<float>();
  size_t block_size = x_shape.SizeFromDimension(axis + 1);
  const float* input = x.template Data<float>();
  uint8_t* output = y.template MutableData<uint8_t>();

  for (size_t n = 0; n < N; n++) {
    const float* current_scale = scale;
    const uint8_t* current_zero_point = zero_point;

    for (size_t bd = 0; bd < static_cast<size_t>(broadcastDim); bd++) {
      float zp = *current_zero_point;
      auto sc = *current_scale;

      for (size_t bs = 0; bs < block_size; bs++) {
        *output++ = static_cast<uint8_t>(clamp(std::round(static_cast<float>(*input++) / sc) + zp, 0.0f, float(UINT8_MAX)));
      }

      current_scale += stride;
      current_zero_point += stride;
    }
  }

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
