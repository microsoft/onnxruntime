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
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(Round, 1, MLFloat16, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()), Round<MLFloat16>);
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(Round, 1, float, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), Round<float>);
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(Round, 1, double, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()), Round<double>);

template <typename T>
Status Round<T>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  auto input = X.template Data<T>();
  auto output = Y.template MutableData<T>();
  auto size = X.Shape().Size();
  for (int64_t i = 0; i < size; ++i, ++output, ++input) {
    *output = ::rint(*input);
  }
  return Status::OK();
}
template <>
Status Round<MLFloat16>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  auto input = X.template Data<MLFloat16>();
  auto output = Y.template MutableData<MLFloat16>();
  auto size = X.Shape().Size();
  for (int64_t i = 0; i < size; ++i, ++output, ++input) {
    *output = MLFloat16(math::floatToHalf(::rint(math::halfToFloat(input->val))));
  }
  return Status::OK();
}

}  // namespace contrib
};  // namespace onnxruntime
