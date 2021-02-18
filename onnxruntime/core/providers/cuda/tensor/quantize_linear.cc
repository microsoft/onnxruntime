// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear.h"
#include "quantize_linear.cuh"

#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

template <class T, class U>
Status QuantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<U>::MappedType CudaU;

  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);

  auto& y = *ctx->Output(0, x.Shape());

  const auto& x_shape = x.Shape();

  const CudaU* input = reinterpret_cast<const CudaU*>(x.template Data<U>());
  T* output = y.template MutableData<T>();

  // TO DO: support per-channel
  ORT_ENFORCE(IsScalarOr1ElementVector(&y_scale), "y_scale must be a scalar or 1D tensor of size 1.");
  ORT_ENFORCE(y_zero_point == nullptr || IsScalarOr1ElementVector(y_zero_point), "y_zero_point must be a scalar or 1D tensor of size 1.");

  const T* zero_point = y_zero_point != nullptr ? y_zero_point->template Data<T>() : nullptr;
  const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.template Data<U>());
  const auto num_of_elements = x_shape.Size();

  CudaQuantizeLinear(Stream(), input, output, scale, zero_point, num_of_elements);

  return Status::OK();
}

template <class T, class U>
Status DequantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<U>::MappedType CudaU;

  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);

  const auto& x_shape = x.Shape();

  auto& y = *ctx->Output(0, x_shape);

  const T* input = x.template Data<T>();
  CudaU* output = reinterpret_cast<CudaU*>(y.template MutableData<U>());

  ORT_ENFORCE(IsScalarOr1ElementVector(&y_scale), "y_scale must be a scalar or 1D tensor of size 1.");
  ORT_ENFORCE(y_zero_point == nullptr || IsScalarOr1ElementVector(y_zero_point), "y_zero_point must be a scalar or 1D tensor of size 1.");

  const T* zero_point = y_zero_point != nullptr ? y_zero_point->template Data<T>() : nullptr;
  const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.template Data<U>());
  const auto num_of_elements = x_shape.Size();

  CudaDequantizeLinear(Stream(), input, output, scale, zero_point, num_of_elements);

  return Status::OK();
}

// register QuantizeLinear kernels
#define REGISTER_Q_KERNEL_TYPED(T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      QuantizeLinear,                                                 \
      kOnnxDomain,                                                    \
      10,                                                             \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T, float>);

REGISTER_Q_KERNEL_TYPED(int8_t)
REGISTER_Q_KERNEL_TYPED(uint8_t)

// register DequantizeLinear kernels
#define REGISTER_DQ_KERNEL_TYPED(T)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      DequantizeLinear,                                           \
      kOnnxDomain,                                                \
      10,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T, float>);

REGISTER_DQ_KERNEL_TYPED(int8_t)
REGISTER_DQ_KERNEL_TYPED(uint8_t)

// specialize QuantizeLinear::ComputeInternal and DequantizeLinear::ComputeInternal
#define SPECIALIZED_QDQ_COMPUTE(T, U)                                                \
  template Status QuantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const; \
  template Status DequantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_QDQ_COMPUTE(int8_t, float)
SPECIALIZED_QDQ_COMPUTE(uint8_t, float)
SPECIALIZED_QDQ_COMPUTE(int8_t, MLFloat16)
SPECIALIZED_QDQ_COMPUTE(uint8_t, MLFloat16)

}  // namespace cuda
}  // namespace onnxruntime
