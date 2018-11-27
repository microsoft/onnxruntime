// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantize_linear.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/tensor/cast_op.h"

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

  TensorShape shape(0, 0);
  std::unique_ptr<Tensor> reshaped_zero_point;
  std::unique_ptr<Tensor> reshaped_scale;

  // if an axis was provided, build the shape necessary for broadcasting across that axis
  if (has_axis_) {
    ONNXRUNTIME_ENFORCE(axis_ < static_cast<int64_t>(x.Shape().NumDimensions()), "axis greater than input data dimension!");
    std::vector<int64_t> shape_;
    shape_.push_back(x_zero_point.Size());
    if (axis_ > 0) {
      for (int64_t i = axis_ - 1; i >= 0; i--) {
        shape_.push_back(1);
      }
    }
    shape = TensorShape(shape_);

    // reshape copies of the inputs for broadcasting.
    TensorAllocator<T> tensorAllocatorUint8(*ctx);
    reshaped_zero_point = tensorAllocatorUint8.Allocate(shape);
    memcpy(reshaped_zero_point->MutableDataRaw(), x_zero_point.DataRaw(), sizeof(T) * x_zero_point.Size());

    TensorAllocator<float> tensorAllocatorFloat(*ctx);
    reshaped_scale = tensorAllocatorFloat.Allocate(shape);
    memcpy(reshaped_scale->MutableDataRaw(), x_scale.DataRaw(), sizeof(float) * x_scale.Size());
  }

  TBroadcaster<T> bc(x, has_axis_ ? *reshaped_zero_point : x_zero_point);
  TBroadcastOutput<float> output(bc.GetSpanSize(), y);
  BroadcastLoop(bc, output,
                [](EigenVectorMap<float> output, T input0, ConstEigenVectorMap<T> input1) {
                  output = (int32_t(input0) - input1.template cast<int32_t>().array()).template cast<float>();
                },
                [](EigenVectorMap<float> output, ConstEigenVectorMap<T> input0, T input1) {
                  output = (input0.template cast<int32_t>().array() - int32_t(input1)).template cast<float>();
                },
                [](EigenVectorMap<float> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) {
                  output = (input0.template cast<int32_t>() - input1.template cast<int32_t>()).template cast<float>();
                });

  TBroadcaster<float> bc2(y, has_axis_ ? *reshaped_scale : x_scale);
  TBroadcastOutput<float> output2(bc2.GetSpanSize(), y);
  BroadcastLoop(bc2, output2,
                [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = input0 * input1.array(); },
                [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = input0.array() * input1; },
                [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = input0.array() * input1.array(); });

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

  TensorShape shape(0, 0);
  std::unique_ptr<Tensor> reshaped_scale;

  TensorAllocator<float> tensorAllocator(*ctx);

  // if an axis was provided, build the shape necessary for broadcasting across that axis
  if (has_axis_) {
    ONNXRUNTIME_ENFORCE(axis_ < static_cast<int64_t>(x.Shape().NumDimensions()), "axis greater than input data dimension!");
    std::vector<int64_t> shape_;
    shape_.push_back(y_zero_point.Size());
    if (axis_ > 0) {
      for (int64_t i = axis_ - 1; i >= 0; i--) {
        shape_.push_back(1);
      }
    }
    shape = TensorShape(shape_);

    reshaped_scale = tensorAllocator.Allocate(shape);
    memcpy(reshaped_scale->MutableDataRaw(), y_scale.DataRaw(), sizeof(float) * y_scale.Size());
  }

  std::unique_ptr<Tensor> W = tensorAllocator.Allocate(x.Shape());
  Tensor* pW = W.get();

  TBroadcaster<float> bc(x, has_axis_ ? *reshaped_scale : y_scale);
  TBroadcastOutput<float> output2(bc.GetSpanSize(), *pW);
  BroadcastLoop(bc, output2,
                [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = (input0 / input1.array()).round(); },
                [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = (input0.array() / input1).round(); },
                [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = (input0.array() / input1.array()).round(); });

  std::unique_ptr<Tensor> Zf = tensorAllocator.Allocate(has_axis_ ? shape : y_zero_point.Shape());
  Tensor* pZf = Zf.get();
  CastData<uint8_t, float>(&y_zero_point, pZf, has_axis_ ? shape : y_zero_point.Shape());

  TBroadcaster<float> bc2(*pW, *pZf);
  TBroadcastOutput<uint8_t> output(bc2.GetSpanSize(), y);
  BroadcastLoop(bc2, output,
                [](EigenVectorMap<uint8_t> output, float input0, ConstEigenVectorMap<float> input1) {
                  for (std::ptrdiff_t i = 0; i < output.size(); i++) {
                    output[i] = uint8_t(clamp(input0 + float(input1[i]), 0.0f, float(UINT8_MAX)));
                  }
                },
                [](EigenVectorMap<uint8_t> output, ConstEigenVectorMap<float> input0, float input1) {
                  for (std::ptrdiff_t i = 0; i < output.size(); i++) {
                    output[i] = uint8_t(clamp(input0[i] + float(input1), 0.0f, float(UINT8_MAX)));
                  }
                },
                [](EigenVectorMap<uint8_t> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) {
                  for (std::ptrdiff_t i = 0; i < output.size(); i++) {
                    output[i] = uint8_t(clamp(input0[i] + float(input1[i]), 0.0f, float(UINT8_MAX)));
                  }
                });

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
