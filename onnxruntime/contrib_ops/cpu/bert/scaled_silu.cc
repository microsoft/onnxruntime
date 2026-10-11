// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/scaled_silu.h"

#include <cmath>

#include "core/platform/threadpool.h"

namespace onnxruntime::contrib {

#define REGISTER_SCALED_SILU_CPU_KERNEL(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      ScaledSiLU, kMSDomain, 1, T, kCpuExecutionProvider,                  \
      KernelDefBuilder()                                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(),      \
                                DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                DataTypeImpl::GetTensorType<BFloat16>()}), \
      ScaledSiLU<T>);

REGISTER_SCALED_SILU_CPU_KERNEL(float)
REGISTER_SCALED_SILU_CPU_KERNEL(MLFloat16)
REGISTER_SCALED_SILU_CPU_KERNEL(BFloat16)

#undef REGISTER_SCALED_SILU_CPU_KERNEL

namespace {

float Sigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + std::exp(-value));
  }
  const float e = std::exp(value);
  return e / (1.0f + e);
}

float LoadFloat(const Tensor& tensor) {
  switch (tensor.GetElementType()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return tensor.Data<float>()[0];
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return static_cast<float>(tensor.Data<MLFloat16>()[0]);
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return static_cast<float>(tensor.Data<BFloat16>()[0]);
    default:
      ORT_THROW("Unsupported scale tensor type.");
  }
}

}  // namespace

template <typename T>
ScaledSiLU<T>::ScaledSiLU(const OpKernelInfo& info)
    : OpKernel(info), alpha_(info.GetAttrOrDefault<float>("alpha", 1.0f)) {}

template <typename T>
Status ScaledSiLU<T>::Compute(OpKernelContext* context) const {
  const auto* x = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(scale == nullptr || scale->Shape().NumDimensions() == 0, "scale must be a scalar");
  const float scale_value = scale == nullptr ? alpha_ : LoadFloat(*scale);
  auto* y = context->Output(0, x->Shape());
  const auto* x_data = x->Data<T>();
  auto* y_data = y->MutableData<T>();
  const int64_t count = x->Shape().Size();
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), onnxruntime::narrow<ptrdiff_t>(count),
      [&](ptrdiff_t i) {
        const T z = static_cast<T>(static_cast<float>(x_data[i]) * scale_value);
        const T sigmoid = static_cast<T>(Sigmoid(static_cast<float>(z)));
        y_data[i] = static_cast<T>(static_cast<float>(z) * static_cast<float>(sigmoid));
      },
      0);
  return Status::OK();
}

template class ScaledSiLU<float>;
template class ScaledSiLU<MLFloat16>;
template class ScaledSiLU<BFloat16>;

}  // namespace onnxruntime::contrib
