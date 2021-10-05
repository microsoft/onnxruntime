// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_gradients.h"
#include "core/providers/common.h"

namespace onnxruntim {

namespace rocm {
ONNX_OPERATOR_KERNEL_EX(
    SigmoidGrad,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SigmoidGrad<float>);

template <typename T>
Status SigmoidGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& Y = *context->Input<Tensor>(1);
  auto& dX = *context->Output(0, dY.Shape());
  EigenVectorArrayMap<float> dx = EigenVectorArrayMap<float>(dX.template MutableData<T>(), dX.Shape().Size());
  ConstEigenVectorArrayMap<float> y = ConstEigenVectorArrayMap<float>(Y.template Data<T>(), Y.Shape().Size());
  ConstEigenVectorArrayMap<float> dy = ConstEigenVectorArrayMap<float>(dY.template Data<T>(), dY.Shape().Size());
  dx = dy * y * (1 -y );
  return Status::OK();
}
} // namespace rocm
} //namespace onnxruntime