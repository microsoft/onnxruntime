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

#define ADD_TYPED_ISNAN_OP(data_type)                                     \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                         \
      IsNaN,                                                              \
      13,                                                                 \
      data_type,                                                          \
      KernelDefBuilder()                                                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<data_type>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),     \
      IsNaN<data_type>);

ADD_TYPED_ISNAN_OP_9(float);
ADD_TYPED_ISNAN_OP_9(double);
ADD_TYPED_ISNAN_OP_9(MLFloat16);
ADD_TYPED_ISNAN_OP(float);
ADD_TYPED_ISNAN_OP(double);
ADD_TYPED_ISNAN_OP(MLFloat16);

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
  if (!X_ptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Null input ptr");
  }
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
}  // namespace onnxruntime
