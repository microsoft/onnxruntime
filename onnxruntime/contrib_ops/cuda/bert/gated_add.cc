// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/gated_add.h"
#include "contrib_ops/cuda/bert/gated_add_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GatedAdd,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GatedAdd<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

template <typename T>
Status GatedAdd<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;

  const Tensor* x = context->Input<Tensor>(0);
  const Tensor* y = context->Input<Tensor>(1);
  const Tensor* gate = context->Input<Tensor>(2);
  const TensorShape& shape = x->Shape();

  ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1, "X must have rank >= 1");
  ORT_RETURN_IF_NOT(y->Shape() == shape, "Y must have the same shape as X");
  ORT_RETURN_IF_NOT(gate->Shape().NumDimensions() == shape.NumDimensions(),
                    "gate must have the same rank as X");

  const size_t last_axis = shape.NumDimensions() - 1;
  const int64_t hidden_size = shape[last_axis];
  ORT_RETURN_IF_NOT(hidden_size > 0, "X last dimension must be positive");
  ORT_RETURN_IF_NOT(gate->Shape()[last_axis] == 1, "gate last dimension must be 1");
  for (size_t axis = 0; axis < last_axis; ++axis) {
    ORT_RETURN_IF_NOT(gate->Shape()[axis] == shape[axis],
                      "gate dimension ", axis, " must match X");
  }

  Tensor* output = context->Output(0, shape);
  return LaunchGatedAddKernel<CudaT>(
      Stream(context),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<const CudaT*>(x->Data<T>()),
      reinterpret_cast<const CudaT*>(y->Data<T>()),
      reinterpret_cast<const CudaT*>(gate->Data<T>()),
      shape.Size(),
      hidden_size);
}

template class GatedAdd<float>;
template class GatedAdd<MLFloat16>;
template class GatedAdd<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime