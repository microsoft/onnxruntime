// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/quantize_linear.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    DequantizeLinear,
    10,
    uint8_t,
    KernelDefBuilder()
        .TypeConstraint("x", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("x_scale", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("x_zero_point", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("y", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<uint8_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    DequantizeLinear,
    10,
    int8_t,
    KernelDefBuilder()
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

  int64_t axis = 0;
  int64_t broadcastDim = x_shape[axis];
  size_t stride = 0;

  if (has_axis_) {
    axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    broadcastDim = x_shape[axis];
    stride = 1;

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(x_scale.Shape().NumDimensions() == 1 && x_scale.Shape().Size() == broadcastDim, "x_scale must be 1D tensor with size ", broadcastDim);
    ORT_ENFORCE(x_zero_point.Shape().NumDimensions() == 1 && x_zero_point.Shape().Size() == broadcastDim, "x_zero_point must be 1D tensor with size ", broadcastDim);
  } else {
    // if no axis, enforce that scale and zero point are scalars
    ORT_ENFORCE(IsScalarOr1ElementVector(&x_scale), "x_scale must be a scalar or 1D tensor or size 1.");
    ORT_ENFORCE(IsScalarOr1ElementVector(&x_zero_point), "x_zero_point must be a scalar or 1D tensor or size 1.");
  }

  size_t N = x_shape.SizeToDimension(axis);
  size_t block_size = x_shape.SizeFromDimension(axis + 1);

  const T* zero_point = x_zero_point.template Data<T>();
  const float* scale = x_scale.template Data<float>();
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

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    QuantizeLinear,
    10,
    uint8_t,
    KernelDefBuilder()
        .TypeConstraint("x", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("y_zero_point", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("y", DataTypeImpl::GetTensorType<uint8_t>()),
    QuantizeLinear<uint8_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    QuantizeLinear,
    10,
    int8_t,
    KernelDefBuilder()
        .TypeConstraint("x", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("y_zero_point", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("y", DataTypeImpl::GetTensorType<int8_t>()),
    QuantizeLinear<int8_t>);

template <typename T>
// formula is Y = X / Scale + ZeroPoint
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto& y_zero_point = *ctx->Input<Tensor>(2);
  auto& y = *ctx->Output(0, x.Shape());
  const auto& x_shape = x.Shape();

  const float* input = x.template Data<float>();
  T* output = y.template MutableData<T>();

  const float qmax = std::numeric_limits<T>::max();
  const float qmin_default = std::numeric_limits<T>::min();
  // adjust qmin for int8 inputs. This is required to keep zero point as zero
  const float qmin = qmin_default == -128 ? -127 : qmin_default;

  // Schema of QuantizeLinearOp changed when it was promoted to onnx domain. In order to maintain backward compatiblity
  // both the versions need to be supported.
  if (ctx->GetOpDomain() != kMSDomain) {
    ORT_ENFORCE(IsScalarOr1ElementVector(&y_scale), "x_scale must be a scalar or 1D tensor or size 1.");
    ORT_ENFORCE(IsScalarOr1ElementVector(&y_zero_point), "x_zero_point must be a scalar or 1D tensor or size 1.");

    const T zero_point = *(y_zero_point.template Data<T>());
    const float scale = *(y_scale.template Data<float>());
    const auto num_of_elements = x_shape.Size();

    MlasQuantizeLinear(input, output, num_of_elements, scale, zero_point);

  } else {
    size_t stride = 0;
    const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    const auto& broadcastDim = x_shape[axis];

    if (has_axis_) {
      // if an axis was specified, ensure the scale and zero point are compatible
      ORT_ENFORCE(y_scale.Shape().NumDimensions() == 1 && y_scale.Shape().Size() == broadcastDim, "x_scale must be 1D tensor with size ", broadcastDim);
      ORT_ENFORCE(y_zero_point.Shape().NumDimensions() == 1 && y_zero_point.Shape().Size() == broadcastDim, "x_zero_point must be 1D tensor with size ", broadcastDim);
      stride = 1;
    } else {
      // if no axis, enforce that scale and zero point are scalars
      ORT_ENFORCE(IsScalarOr1ElementVector(&y_scale), "x_scale must be a scalar or 1D tensor or size 1.");
      ORT_ENFORCE(IsScalarOr1ElementVector(&y_zero_point), "x_zero_point must be a scalar or 1D tensor or size 1.");
    }

    size_t N = x_shape.SizeToDimension(axis);
    size_t block_size = x_shape.SizeFromDimension(axis + 1);
    const T* zero_point = y_zero_point.template Data<T>();
    const float* scale = y_scale.template Data<float>();

    for (size_t n = 0; n < N; n++) {
      const float* current_scale = scale;
      const T* current_zero_point = zero_point;

      for (size_t bd = 0; bd < static_cast<size_t>(broadcastDim); bd++) {
        float zp = *current_zero_point;
        auto sc = *current_scale;

        for (size_t bs = 0; bs < block_size; bs++) {
          *output++ = static_cast<T>(clamp(std::round(static_cast<float>(*input++) / sc) + zp, qmin, qmax));
        }

        current_scale += stride;
        current_zero_point += stride;
      }
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
