// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "round.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/data_types.h"
#include <cmath>
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(Round, 11, 21, MLFloat16, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()), Round<MLFloat16>);
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(Round, 11, 21, float, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), Round<float>);
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(Round, 11, 21, double, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()), Round<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(Round, 22, MLFloat16, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()), Round<MLFloat16>);
ONNX_CPU_OPERATOR_TYPED_KERNEL(Round, 22, float, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), Round<float>);
ONNX_CPU_OPERATOR_TYPED_KERNEL(Round, 22, double, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()), Round<double>);

template <typename T>
Status Round<T>::Compute(OpKernelContext* ctx) const {
  const auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  const auto* input = X.Data<T>();
  auto* output = Y.MutableData<T>();
  const auto size = narrow<Eigen::Index>(X.Shape().Size());

  EigenArrayMap<T> Y_arr(output, 1, size);
  ConstEigenArrayMap<T> X_arr(input, 1, size);
  Y_arr = X_arr.rint();

  return Status::OK();
}
template <>
Status Round<MLFloat16>::Compute(OpKernelContext* ctx) const {
  const auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  const auto* input = X.Data<MLFloat16>();
  auto* output = Y.MutableData<MLFloat16>();
  const auto size = narrow<Eigen::Index>(X.Shape().Size());

  ConstEigenArrayMap<Eigen::half> X_arr(reinterpret_cast<const Eigen::half*>(input), 1, size);
  EigenArrayMap<Eigen::half> Y_arr(reinterpret_cast<Eigen::half*>(output), 1, size);
  Y_arr = X_arr.rint();

  return Status::OK();
}

};  // namespace onnxruntime
