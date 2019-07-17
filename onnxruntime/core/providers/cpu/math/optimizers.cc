// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/optimizers.h"

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "gsl/gsl_util"

namespace onnxruntime {

template <typename T>
Status SGDOptimizer<T>::Compute(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  Tensor& W = *ctx->MutableInput<Tensor>(1);
  const Tensor& G = *ctx->Input<Tensor>(2);
  const TensorShape WeightShape{W.Shape()};
  Tensor& NW = *ctx->Output(0, WeightShape);

  int N = gsl::narrow_cast<int>(WeightShape.Size());
  T* mutable_new_weight = NW.template MutableData<T>();
  const T* new_weight = NW.template Data<T>();
  math::Scale<T, CPUMathUtil>(N, *ETA.template Data<float>(), G.template Data<T>(), mutable_new_weight, nullptr);
  math::Sub<T, CPUMathUtil>(N, W.template Data<T>(), new_weight, mutable_new_weight, nullptr);
  memcpy(W.template MutableData<T>(), new_weight, W.Size());
  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    SGDOptimizer,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SGDOptimizer<float>);

}  // namespace onnxruntime
