// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/quantize_linear.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

#define REGISTER_DEQUANTIZELINEAR(T)                              \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                 \
      DequantizeLinear,                                           \
      13,                                                         \
      T,                                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);

#define REGISTER_DEQUANTIZELINEAR_VERSIONED(T)                    \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                       \
      DequantizeLinear,                                           \
      10,                                                         \
      12,                                                         \
      T,                                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);

REGISTER_DEQUANTIZELINEAR(int8_t)
REGISTER_DEQUANTIZELINEAR(uint8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint8_t)

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

  if (IsScalarOr1ElementVector(&x_scale)) {  // per-tensor DequantizeLinear
    N = 1;
    broadcast_dim = 1;
    block_size = static_cast<size_t>(x_shape.Size());

    // enforce that zero point are scalars
    ORT_ENFORCE(x_zero_point == nullptr || IsScalarOr1ElementVector(x_zero_point),
                "x_zero_point must be null or a scalar or 1D tensor or size 1.");
  } else {  // per-channel DequantizeLinear
    const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    N = x_shape.SizeToDimension(axis);
    broadcast_dim = x_shape[axis];
    block_size = x_shape.SizeFromDimension(axis + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(x_scale.Shape().NumDimensions() == 1 && x_scale.Shape().Size() == broadcast_dim,
                "x_scale must be 1D tensor with size ",
                broadcast_dim);
    ORT_ENFORCE(x_zero_point == nullptr || (x_zero_point->Shape().NumDimensions() == 1 && x_zero_point->Shape().Size() == broadcast_dim),
                "x_zero_point must be nulll or 1D tensor with size ",
                broadcast_dim);
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

#define REGISTER_QUANTIZELINEAR(T)                                    \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                     \
      QuantizeLinear,                                                 \
      13,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_VERSIONED(T)                          \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                           \
      QuantizeLinear,                                                 \
      10,                                                             \
      12,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T>);

REGISTER_QUANTIZELINEAR(int8_t)
REGISTER_QUANTIZELINEAR(uint8_t)
REGISTER_QUANTIZELINEAR_VERSIONED(int8_t)
REGISTER_QUANTIZELINEAR_VERSIONED(uint8_t)

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

  if (IsScalarOr1ElementVector(&y_scale)) {  // per-tensor QuantizeLinear
    N = 1;
    broadcast_dim = 1;
    block_size = x_shape.Size();

    // enforce zero point are scalars
    ORT_ENFORCE(y_zero_point == nullptr || IsScalarOr1ElementVector(y_zero_point),
                "y_zero_point must be a scalar or 1D tensor or size 1, same as the y_scale.");
  } else {  // per-channel QuantizeLinear
    const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    N = x_shape.SizeToDimension(axis);
    broadcast_dim = x_shape[axis];
    block_size = x_shape.SizeFromDimension(axis + 1);

    // ensure the scale and zero point are compatible
    ORT_ENFORCE(y_scale.Shape().NumDimensions() == 1 && y_scale.Shape().Size() == broadcast_dim,
                "y_scale must be 1D tensor with size ",
                broadcast_dim);
    ORT_ENFORCE(y_zero_point == nullptr || (y_zero_point->Shape().NumDimensions() == 1 && y_zero_point->Shape().Size() == broadcast_dim),
                "y_zero_point must be null or 1D tensor with size ",
                broadcast_dim);
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
