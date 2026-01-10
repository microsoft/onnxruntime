// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/isnan.h"

#include "core/common/common.h"
#include "core/framework/math.h"
#include "core/framework/tensor.h"
#include "Eigen/src/Core/arch/Default/Half.h"

namespace onnxruntime {
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsNaN
#define ADD_TYPED_ISNAN_OP_9(data_type)                                   \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                               \
      IsNaN,                                                              \
      9, 12,                                                              \
      data_type,                                                          \
      KernelDefBuilder()                                                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<data_type>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),     \
      IsNaN<data_type>);

#define ADD_TYPED_ISNAN_OP_13(data_type)                                  \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                               \
      IsNaN,                                                              \
      13, 19,                                                             \
      data_type,                                                          \
      KernelDefBuilder()                                                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<data_type>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),     \
      IsNaN<data_type>);

#define ADD_TYPED_ISNAN_OP(data_type)                                     \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                         \
      IsNaN,                                                              \
      20,                                                                 \
      data_type,                                                          \
      KernelDefBuilder()                                                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<data_type>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),     \
      IsNaN<data_type>);

ADD_TYPED_ISNAN_OP_9(float);
ADD_TYPED_ISNAN_OP_9(double);
ADD_TYPED_ISNAN_OP_9(MLFloat16);
ADD_TYPED_ISNAN_OP_13(float);
ADD_TYPED_ISNAN_OP_13(double);
ADD_TYPED_ISNAN_OP_13(MLFloat16);
ADD_TYPED_ISNAN_OP_13(BFloat16);
ADD_TYPED_ISNAN_OP(float);
ADD_TYPED_ISNAN_OP(double);
ADD_TYPED_ISNAN_OP(MLFloat16);
ADD_TYPED_ISNAN_OP(BFloat16);

#if !defined(DISABLE_FLOAT8_TYPES)
ADD_TYPED_ISNAN_OP(Float8E4M3FN);
ADD_TYPED_ISNAN_OP(Float8E4M3FNUZ);
ADD_TYPED_ISNAN_OP(Float8E5M2);
ADD_TYPED_ISNAN_OP(Float8E5M2FNUZ);
#endif

template <typename T>
Status IsNaN<T>::Compute(OpKernelContext* context) const {
  const auto* X_ptr = context->Input<Tensor>(0);
  if (!X_ptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Null input ptr");
  }
  auto& X = *X_ptr;
  auto& dims = X.Shape();
  auto& Y = *context->Output(0, dims);

  EigenMap<bool>(Y) = EigenMap<T>(X).array().isNaN();

  return Status::OK();
}

template <>
Status IsNaN<MLFloat16>::Compute(OpKernelContext* context) const {
  const auto* X_ptr = context->Input<Tensor>(0);

  auto X_data = X_ptr->Data<MLFloat16>();
  auto& dims = X_ptr->Shape();
  auto shape_size = dims.Size();
  auto& Y = *context->Output(0, dims);

  EigenMap<bool>(Y) =
      ConstEigenVectorMap<Eigen::half>(static_cast<const Eigen::half*>(static_cast<const void*>(X_data)), onnxruntime::narrow<size_t>(shape_size))
          .array()
          .isNaN();

  return Status::OK();
}

template <>
Status IsNaN<BFloat16>::Compute(OpKernelContext* context) const {
  const auto* X_ptr = context->Input<Tensor>(0);

  auto X_data = X_ptr->DataAsSpan<BFloat16>();
  auto& Y = *context->Output(0, X_ptr->Shape());

  std::transform(X_data.begin(), X_data.end(), Y.MutableData<bool>(),
                 [](BFloat16 x) { return x.IsNaN(); });

  return Status::OK();
}

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
Status IsNaN<Float8E4M3FN>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  auto& dims = X->Shape();
  auto& Y = *context->Output(0, dims);

  auto input = ConstEigenVectorMap<uint8_t>(static_cast<const uint8_t*>(static_cast<const void*>(X->Data<Float8E4M3FN>())), onnxruntime::narrow<size_t>(dims.Size()));
  auto output = EigenMap<bool>(Y);

  // S.1111.111
  std::transform(input.begin(), input.end(), output.begin(), [](uint8_t c) { return (c & 0x7f) == 0x7f; });
  return Status::OK();
}

template <>
Status IsNaN<Float8E4M3FNUZ>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  auto X_data = X->Data<Float8E4M3FNUZ>();
  auto& dims = X->Shape();
  auto shape_size = dims.Size();
  auto& Y = *context->Output(0, dims);

  // 1.0000.000
  EigenMap<bool>(Y) =
      ConstEigenVectorMap<uint8_t>(static_cast<const uint8_t*>(static_cast<const void*>(X_data)), onnxruntime::narrow<size_t>(shape_size)).array() == 0x80;

  return Status::OK();
}

template <>
Status IsNaN<Float8E5M2>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  auto& dims = X->Shape();
  auto& Y = *context->Output(0, dims);

  auto input = ConstEigenVectorMap<uint8_t>(static_cast<const uint8_t*>(static_cast<const void*>(X->Data<Float8E5M2>())), onnxruntime::narrow<size_t>(dims.Size()));
  auto output = EigenMap<bool>(Y);

  // S.11111.{01, 10, 11}
  std::transform(input.begin(), input.end(), output.begin(), [](uint8_t c) { return ((c & 0x7c) == 0x7c) && ((c & 0x03) != 0x00); });
  return Status::OK();
}

template <>
Status IsNaN<Float8E5M2FNUZ>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  auto X_data = X->Data<Float8E5M2FNUZ>();
  auto& dims = X->Shape();
  auto shape_size = dims.Size();
  auto& Y = *context->Output(0, dims);

  // 1.0000.000
  EigenMap<bool>(Y) = ConstEigenVectorMap<uint8_t>(static_cast<const uint8_t*>(static_cast<const void*>(X_data)), onnxruntime::narrow<size_t>(shape_size)).array() == 0x80;

  return Status::OK();
}
#endif
}  // namespace onnxruntime
