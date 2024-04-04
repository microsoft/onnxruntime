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

namespace onnxruntime {

ONNX_CPU_OPERATOR_TYPED_KERNEL(Round, 11, MLFloat16, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()), Round<MLFloat16>);
ONNX_CPU_OPERATOR_TYPED_KERNEL(Round, 11, float, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), Round<float>);
ONNX_CPU_OPERATOR_TYPED_KERNEL(Round, 11, double, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()), Round<double>);

template <typename T>
Status Round<T>::Compute(OpKernelContext* ctx) const {
  const auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  auto* input = X.Data<T>();
  auto* output = Y.MutableData<T>();
  const auto size = X.Shape().Size();
  for (int64_t i = 0; i < size; ++i, ++output, ++input) {
    *output = ::rint(*input);
  }
  return Status::OK();
}
template <>
Status Round<MLFloat16>::Compute(OpKernelContext* ctx) const {
  const auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  auto* input = X.Data<MLFloat16>();
  auto* output = Y.MutableData<MLFloat16>();
  const auto size = X.Shape().Size();
  for (int64_t i = 0; i < size; ++i, ++output, ++input) {
    *output = MLFloat16(static_cast<float>(::rint(input->ToFloat())));
  }
  return Status::OK();
}

};  // namespace onnxruntime
