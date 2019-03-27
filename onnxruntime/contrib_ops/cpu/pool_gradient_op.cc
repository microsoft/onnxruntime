// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pool_gradient_op.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_KERNEL(
    MaxPoolGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MaxPoolGrad<float>);

template <typename T>
Status MaxPoolGrad<T>::Compute(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* indices = context->Input<Tensor>(1);
  ORT_ENFORCE(dY->Shape() == indices->Shape(), "The shape of dY and indices does not match in MaxPoolGrad.");

  const TensorShape& dX_shape = TensorShape::ReinterpretBaseType(output_tensor_shapes_[0]);
  Tensor* dX = context->Output(0, dX_shape);

  const float* dY_data = dY->template Data<float>();
  const int64_t* indices_data = indices->template Data<int64_t>();
  float* dX_data = dX->template MutableData<float>();

  EigenVectorMap<T>(dX_data, dX_shape.Size()).setZero();

  for (int64_t i = 0; i < dY->Shape().Size(); ++i) {
    float* p_dX_data = dX_data + indices_data[i];
    *p_dX_data += dY_data[i];
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
