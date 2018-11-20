// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "isnan.h"
#include "onnx/defs/schema.h"
#include "core/util/math_cpuonly.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    IsNaN,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),
    contrib::IsNaN<float>);

template <>
Status IsNaN<float>::Compute(OpKernelContext* context) const {
  const Tensor* X_ptr = context->Input<Tensor>(0);
  if (!X_ptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Null input ptr");
  }
  auto& X = *X_ptr;
  auto& dims = X.Shape();
  auto& Y = *context->Output(0, dims);

  EigenMap<bool>(Y) = EigenMap<float>(X).array().isNaN();

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
