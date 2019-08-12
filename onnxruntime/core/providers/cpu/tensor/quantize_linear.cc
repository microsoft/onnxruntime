// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/quantize_linear.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/common.h"
#include <cmath>
#include <cfenv>

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
  const auto& scale_shape = x_scale.Shape();
  const auto& zero_point_shape = x_zero_point.Shape();

  // enforce that scale and zero point are scalars or 1D tensors with size == 1
  ORT_ENFORCE(scale_shape.NumDimensions() == 0 || (scale_shape.NumDimensions() == 1 && scale_shape.GetDims().size() == 1), "x_scale must be a scalar.");
  ORT_ENFORCE(zero_point_shape.NumDimensions() == 0 || (zero_point_shape.NumDimensions() == 1 && zero_point_shape.GetDims().size() == 1), "x_zero_point must be a scalar.");

  const T zero_point = *(x_zero_point.template Data<T>());
  const float scale = *(x_scale.template Data<float>());
  const T* input = x.template Data<T>();
  auto* output = y.template MutableData<float>();
  const auto num_of_elements = x_shape.Size();

  for (int i = 0; i < num_of_elements; ++i) {
    output[i] = static_cast<float>(static_cast<const int>(input[i]) - zero_point) * scale;
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

// clamp doesn't exist in the version of <algorithm> that we're using, so
// make a local one.
static float clamp(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static float RoundHalfToEven(float input) {
  std::fesetround(FE_TONEAREST);
  auto result = std::nearbyintf(input);
  return result;
}

template <typename T>
// formula is Y = X / Scale + ZeroPoint
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto& y_zero_point = *ctx->Input<Tensor>(2);
  auto& y = *ctx->Output(0, x.Shape());

  const auto& x_shape = x.Shape();
  const auto& scale_shape = y_scale.Shape();
  const auto& zero_point_shape = y_zero_point.Shape();
  

  //enforce that scale and zero point are scalars or 1D tensors with size == 1
  ORT_ENFORCE(scale_shape.NumDimensions() == 0 || (scale_shape.NumDimensions() == 1 && scale_shape.GetDims().size() == 1), "x_scale must be a scalar.");
  ORT_ENFORCE(zero_point_shape.NumDimensions() == 0 || (zero_point_shape.NumDimensions() == 1 && zero_point_shape.GetDims().size() == 1), "x_zero_point must be a scalar.");
  
  const T zero_point = *(y_zero_point.template Data<T>());
  const float scale = *(y_scale.template Data<float>());
  const auto* input = x.template Data<float>();
  auto* output = y.template MutableData<T>();
  const auto num_of_elements = x_shape.Size();
  const float qmax = std::numeric_limits<T>::max();
  const float qmin_default = std::numeric_limits<T>::min();
  // adjust qmin for int8 inputs. This is required to keep zero point as zero
  const float qmin = qmin_default == -128 ? -127 : qmin_default;

  for (int i = 0; i < num_of_elements; ++i) {
    output[i] = static_cast<T>(clamp(RoundHalfToEven(static_cast<float>(input[i]/scale)) + zero_point, qmin, qmax));
  }

  return Status::OK();
}
}  // namespace onnxruntime
