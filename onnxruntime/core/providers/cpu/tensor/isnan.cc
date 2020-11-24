// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "isnan.h"
#include "Eigen/Core"
#include "Eigen/Dense"

#include "core/util/math_cpuonly.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "Eigen/src/Core/arch/Default/Half.h"

namespace onnxruntime {
template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
auto EigenMap(Tensor& t) -> EigenVectorMap<T> {
  return EigenVectorMap<T>(t.template MutableData<T>(), t.Shape().Size());
}
template <typename T>
auto EigenMap(const Tensor& t) -> ConstEigenVectorMap<T> {
  return ConstEigenVectorMap<T>(t.template Data<T>(), t.Shape().Size());
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#IsNaN
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
ADD_TYPED_ISNAN_OP_9(MLFloat16);
ADD_TYPED_ISNAN_OP(float);
ADD_TYPED_ISNAN_OP(MLFloat16);

template <>
Status IsNaN<float>::Compute(OpKernelContext* context) const {
  const auto* X_ptr = context->Input<Tensor>(0);
  if (!X_ptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Null input ptr");
  }
  auto& X = *X_ptr;
  auto& dims = X.Shape();
  auto& Y = *context->Output(0, dims);

  EigenMap<bool>(Y) = EigenMap<float>(X).array().isNaN();

  return Status::OK();
}

template <>
Status IsNaN<MLFloat16>::Compute(OpKernelContext* context) const {
  const auto* X_ptr = context->Input<Tensor>(0);
  if (!X_ptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Null input ptr");
  }
  auto X_data = X_ptr->template Data<MLFloat16>();
  auto& dims = X_ptr->Shape();
  auto shape_size = dims.Size();
  auto& Y = *context->Output(0, dims);

  EigenMap<bool>(Y) =
      ConstEigenVectorMap<Eigen::half>(static_cast<const Eigen::half*>(static_cast<const void*>(X_data)), shape_size)
          .array()
          .isNaN();

  return Status::OK();
}
}  // namespace onnxruntime
