// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear.h"
#include "quantize_linear.cuh"

#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                      \
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

REGISTER_KERNEL_TYPED(int8_t)
REGISTER_KERNEL_TYPED(uint8_t)

template <class T, class U>
Status QuantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<U>::MappedType CudaU;

  auto x = ctx->Input<Tensor>(0);
  auto y_scale = ctx->Input<Tensor>(1);
  auto y_zero_point = ctx->Input<Tensor>(2);
  ORT_ENFORCE(x != nullptr &&
              y_scale != nullptr &&
              y_zero_point != nullptr);
  auto y = ctx->Output(0, x->Shape());
  ORT_ENFORCE(y != nullptr);

  const auto& x_shape = x->Shape();

  const CudaU* input = reinterpret_cast<const CudaU*>(x->template Data<U>());
  T* output = y->template MutableData<T>();

  // TO DO: support per-channel
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale), "x_scale must be a scalar or 1D tensor of size 1.");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_zero_point), "x_zero_point must be a scalar or 1D tensor of size 1.");

  const T* zero_point = y_zero_point->template Data<T>();
  const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale->template Data<U>());
  const auto num_of_elements = x_shape.Size();

  CudaQuantizeLinear(input, output, scale, zero_point, num_of_elements);

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(DequantizeLinear,
                              kOnnxDomain,
                              10,
                              uint8_t,
                              kCudaExecutionProvider,
                              KernelDefBuilder()
                                  .TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
                              DequantizeLinear<uint8_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(DequantizeLinear,
                              kOnnxDomain,
                              10,
                              int8_t,
                              kCudaExecutionProvider,
                              KernelDefBuilder()
                                  .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
                              DequantizeLinear<int8_t>);

template <class T>
Status DequantizeLinear<T>::ComputeInternal(OpKernelContext* ctx) const {
  auto x = ctx->Input<Tensor>(0);
  auto y_scale = ctx->Input<Tensor>(1);
  auto y_zero_point = ctx->Input<Tensor>(2);
  ORT_ENFORCE(x != nullptr &&
              y_scale != nullptr &&
              y_zero_point != nullptr);

  const auto& x_shape = x->Shape();

  auto y = ctx->Output(0, x_shape);
  ORT_ENFORCE(y != nullptr);

  const T* input = x->template Data<T>();
  float* output = y->template MutableData<float>();

  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale), "x_scale must be a scalar or 1D tensor of size 1.");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_zero_point), "x_zero_point must be a scalar or 1D tensor of size 1.");

  const T* zero_point = y_zero_point->template Data<T>();
  const float* scale = y_scale->template Data<float>();
  const auto num_of_elements = x_shape.Size();

  CudaDequantizeLinear(input, output, scale, zero_point, num_of_elements);

  return Status::OK();
}

#define SPECIALIZED_QL_COMPUTE(T, U) \
  template Status QuantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_QL_COMPUTE(int8_t, float)
SPECIALIZED_QL_COMPUTE(uint8_t, float)
SPECIALIZED_QL_COMPUTE(int8_t, MLFloat16)
SPECIALIZED_QL_COMPUTE(uint8_t, MLFloat16)

}  // namespace cuda
}  // namespace onnxruntime
