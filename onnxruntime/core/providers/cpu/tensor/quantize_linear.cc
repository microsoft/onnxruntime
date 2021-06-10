// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/quantize_linear.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/qmath.h"

namespace onnxruntime {

static void PrepareForQDQ(const TensorShape& input_shape,
                          const Tensor& scale,
                          const Tensor* zero_point_ptr,
                          int64_t axis,
                          int64_t& block_count,
                          int64_t& broadcast_dim,
                          int64_t& block_size) {
  if (IsScalarOr1ElementVector(&scale)) {  // per-tensor QuantizeLinear/DequantizeLinear
    block_count = 1;
    broadcast_dim = 1;
    block_size = static_cast<size_t>(input_shape.Size());

    // enforce that zero point are scalars
    ORT_ENFORCE(zero_point_ptr == nullptr || IsScalarOr1ElementVector(zero_point_ptr),
                "x_zero_point must be null or a scalar or 1D tensor or size 1.");
  } else {  // per-channel QuantizeLinear/DequantizeLinear
    const int64_t axis_no_neg = HandleNegativeAxis(axis, input_shape.NumDimensions());
    block_count = input_shape.SizeToDimension(axis_no_neg);
    broadcast_dim = input_shape[axis_no_neg];
    block_size = input_shape.SizeFromDimension(axis_no_neg + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(scale.Shape().NumDimensions() == 1 && scale.Shape()[0] == broadcast_dim,
                "scale must be 1D tensor with size ",
                broadcast_dim);
    ORT_ENFORCE(zero_point_ptr == nullptr || (zero_point_ptr->Shape().NumDimensions() == 1 && zero_point_ptr->Shape()[0] == broadcast_dim),
                "x_zero_point must be null or 1D tensor with size ",
                broadcast_dim);
  }
}

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
REGISTER_DEQUANTIZELINEAR(int32_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int32_t)

// formula is Y = (X - ZeroPoint) * Scale
template <typename T>
Status DequantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& x_scale = *ctx->Input<Tensor>(1);
  auto* x_zero_point = ctx->Input<Tensor>(2);

  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;

  PrepareForQDQ(x.Shape(), x_scale, x_zero_point, axis_, N, broadcast_dim, block_size);

  const float* scale = x_scale.template Data<float>();
  const T* input = x.template Data<T>();
  float* output = y.template MutableData<float>();

  const T* zero_point = x_zero_point ? x_zero_point->template Data<T>() : nullptr;
  if (std::is_same<T, int32_t>::value) {
    ORT_ENFORCE(zero_point == nullptr ||
                    std::all_of(zero_point,
                                zero_point + x_zero_point->Shape().Size(),
                                [](int32_t zp) { return zp == 0; }),
                "DequantizeLinear with type int32 should have no zero point or all zero points should be 0");
  }

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

// formula is Y = X / Scale + ZeroPoint
template <typename T>
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;
  PrepareForQDQ(x.Shape(), y_scale, y_zero_point, axis_, N, broadcast_dim, block_size);

  const T* zero_point = y_zero_point != nullptr ? y_zero_point->template Data<T>() : nullptr;
  const float* scale = y_scale.template Data<float>();
  const float* input = x.template Data<float>();
  T* output = y.template MutableData<T>();

  for (size_t n = 0; n < static_cast<size_t>(N); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      T zp = zero_point != nullptr ? zero_point[bd] : 0;
      ParQuantizeLinear(input, output, static_cast<size_t>(block_size), scale[bd], zp, ctx->GetOperatorThreadPool());
      input += block_size;
      output += block_size;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
