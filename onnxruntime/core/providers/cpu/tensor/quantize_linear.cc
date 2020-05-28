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
        .TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
    DequantizeLinear<uint8_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    DequantizeLinear,
    10,
    int8_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
    DequantizeLinear<int8_t>);

template <typename T>
// formula is Y = (X - ZeroPoint) * Scale
Status DequantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& x_scale = *ctx->Input<Tensor>(1);
  auto* x_zero_point = ctx->Input<Tensor>(2);

  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;

  if (has_axis_) {  // custom DequantizeLinear only
    const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    N = x_shape.SizeToDimension(axis);
    broadcast_dim = x_shape[axis];
    block_size = x_shape.SizeFromDimension(axis + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(x_scale.Shape().NumDimensions() == 1 && x_scale.Shape().Size() == broadcast_dim,
                "x_scale must be 1D tensor with size ",
                broadcast_dim);
    ORT_ENFORCE(x_zero_point != nullptr && x_zero_point->Shape().NumDimensions() == 1 && x_zero_point->Shape().Size() == broadcast_dim,
                "x_zero_point must be 1D tensor with size ",
                broadcast_dim);
  } else {
    N = 1;
    broadcast_dim = 1;
    block_size = static_cast<size_t>(x_shape.Size());

    // if no axis, enforce that scale and zero point are scalars
    ORT_ENFORCE(IsScalarOr1ElementVector(&x_scale), "x_scale must be a scalar or 1D tensor or size 1.");
    ORT_ENFORCE(x_zero_point == nullptr || IsScalarOr1ElementVector(x_zero_point), "x_zero_point must be a scalar or 1D tensor or size 1.");
  }

  const T* zero_point = x_zero_point ? x_zero_point->template Data<T>() : nullptr;
  const float* scale = x_scale.template Data<float>();
  const T* input = x.template Data<T>();
  float* output = y.template MutableData<float>();

  for (size_t n = 0; n < static_cast<size_t>(N); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      auto zp = zero_point ? static_cast<int32_t>(zero_point[bd]) : 0;
      auto sc = scale[bd];

      for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++) {
        *output++ = static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc;
      }
    }
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    QuantizeLinear,
    10,
    uint8_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    QuantizeLinear<uint8_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    QuantizeLinear,
    10,
    int8_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>()),
    QuantizeLinear<int8_t>);

template <typename T>
// formula is Y = X / Scale + ZeroPoint
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;

  if (has_axis_) {  // custom QuantizeLinear only
    const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    N = x_shape.SizeToDimension(axis);
    broadcast_dim = x_shape[axis];
    block_size = x_shape.SizeFromDimension(axis + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(y_scale.Shape().NumDimensions() == 1 && y_scale.Shape().Size() == broadcast_dim,
                "y_scale must be 1D tensor with size ",
                broadcast_dim);
    ORT_ENFORCE(y_zero_point != nullptr && y_zero_point->Shape().NumDimensions() == 1 && y_zero_point->Shape().Size() == broadcast_dim,
                "y_zero_point must be 1D tensor with size ",
                broadcast_dim);
  } else {
    N = 1;
    broadcast_dim = 1;
    block_size = x_shape.Size();

    // if no axis, enforce that scale and zero point are scalars
    ORT_ENFORCE(IsScalarOr1ElementVector(&y_scale),
                "y_scale must be a scalar or 1D tensor or size 1.");
    ORT_ENFORCE(y_zero_point == nullptr || IsScalarOr1ElementVector(y_zero_point),
                "y_zero_point must be a scalar or 1D tensor or size 1.");
  }

  const T* zero_point = y_zero_point != nullptr ? y_zero_point->template Data<T>() : nullptr;
  const float* scale = y_scale.template Data<float>();
  const float* input = x.template Data<float>();
  T* output = y.template MutableData<T>();

  for (size_t n = 0; n < static_cast<size_t>(N); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      T zp = zero_point != nullptr ? zero_point[bd] : 0;
      MlasQuantizeLinear(input, output, static_cast<size_t>(block_size), scale[bd], zp);
      input += block_size;
      output += block_size;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
