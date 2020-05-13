// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear.h"

#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"

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
  auto& x_zero_point = *ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  const T* zero_point = x_zero_point.template Data<T>();
  const float* scale = x_scale.template Data<float>();
  const T* input = x.template Data<T>();
  float* output = y.template MutableData<float>();
  concurrency::ThreadPool* tp = ctx->GetOperatorThreadPool();

  if (has_axis_) {
    const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    int64_t N = x_shape.SizeToDimension(axis);
    int64_t broadcast_dim = x_shape[axis];
    int64_t block_size = x_shape.SizeFromDimension(axis + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(x_scale.Shape().NumDimensions() == 1 && x_scale.Shape().Size() == broadcast_dim, "x_scale must be 1D tensor with size ", broadcast_dim);
    ORT_ENFORCE(x_zero_point.Shape().NumDimensions() == 1 && x_zero_point.Shape().Size() == broadcast_dim, "x_zero_point must be 1D tensor with size ", broadcast_dim);

    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(N * broadcast_dim),
        [&](ptrdiff_t task_idx) {
          const T* input_tmp = input + task_idx * block_size;
          float* output_tmp = output + task_idx * block_size;

          int32_t zp = static_cast<int32_t>(zero_point[task_idx % broadcast_dim]);
          float sc = scale[task_idx % broadcast_dim];
          for (int64_t idx = 0; idx < block_size; idx++) {
            *output_tmp++ = static_cast<float>(static_cast<int32_t>(*input_tmp++) - zp) * sc;
          }
        },
        0);

  } else {
    // if no axis, enforce that scale and zero point are scalars
    ORT_ENFORCE(IsScalarOr1ElementVector(&x_scale), "x_scale must be a scalar or 1D tensor or size 1.");
    ORT_ENFORCE(IsScalarOr1ElementVector(&x_zero_point), "x_zero_point must be a scalar or 1D tensor or size 1.");

    concurrency::ThreadPool::TryParallelFor(tp, x_shape.Size(), 2.0 /*cost*/, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      const T* input_tmp = input + begin;
      float* output_tmp = output + begin;
      int32_t zp = static_cast<int32_t>(*zero_point);
      float sc = *scale;
      for (; output_tmp != output + end;) {
        *output_tmp++ = static_cast<float>(static_cast<int32_t>(*input_tmp++) - zp) * sc;
      }
    });
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
  auto& y_zero_point = *ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  const T* zero_point = y_zero_point.template Data<T>();
  const float* scale = y_scale.template Data<float>();
  const float* input = x.template Data<float>();
  T* output = y.template MutableData<T>();
  concurrency::ThreadPool* tp = ctx->GetOperatorThreadPool();

  if (has_axis_) {
    const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    int64_t N = x_shape.SizeToDimension(axis);
    int64_t broadcast_dim = x_shape[axis];
    int64_t block_size = x_shape.SizeFromDimension(axis + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(y_scale.Shape().NumDimensions() == 1 && y_scale.Shape().Size() == broadcast_dim, "x_scale must be 1D tensor with size ", broadcast_dim);
    ORT_ENFORCE(y_zero_point.Shape().NumDimensions() == 1 && y_zero_point.Shape().Size() == broadcast_dim, "x_zero_point must be 1D tensor with size ", broadcast_dim);

    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(N * broadcast_dim),
        [&](ptrdiff_t task_idx) {
          MlasQuantizeLinear(input + task_idx * block_size,
                             output + task_idx * block_size,
                             static_cast<size_t>(block_size),
                             scale[task_idx % broadcast_dim],
                             zero_point[task_idx % broadcast_dim]);
        },
        0);
  } else {
    // if no axis, enforce that scale and zero point are scalars
    ORT_ENFORCE(IsScalarOr1ElementVector(&y_scale), "x_scale must be a scalar or 1D tensor or size 1.");
    ORT_ENFORCE(IsScalarOr1ElementVector(&y_zero_point), "x_zero_point must be a scalar or 1D tensor or size 1.");

    concurrency::ThreadPool::TryParallelFor(tp, x_shape.Size(), 2.0 /*cost*/, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      MlasQuantizeLinear(input + begin, output + begin, end - begin, *scale, *zero_point);
    });
  }

  return Status::OK();
}

}  // namespace onnxruntime
