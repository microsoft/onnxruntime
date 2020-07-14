// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/nn/broadcast_grad_args.h"

namespace onnxruntime {
namespace contrib {
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      BroadcastGradientArgs,                                      \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BroadcastGradientArgs<T>);

REGISTER_KERNEL_TYPED(int64_t)

template <typename T>
Status BroadcastGradientArgs<T>::Compute(OpKernelContext* context) const {
  const Tensor* a_tensor = context->Input<Tensor>(0);
  const Tensor* b_tensor = context->Input<Tensor>(1);
  const T* A_dims = a_tensor->template Data<T>();
  const T* B_dims = b_tensor->template Data<T>();

  const int a_size = a_tensor->Shape().Size();
  const int b_size = b_tensor->Shape().Size();

  int ndim = int(std::max(a_size, b_size));
  const TensorShape op_shape = {1, ndim};
  Tensor* A_axes = context->Output(0, op_shape);
  T* A_axes_data = A_axes->template MutableData<T>();
  Tensor* B_axes = context->Output(1, op_shape);
  T* B_axes_data = B_axes->template MutableData<T>();
  if (!A_axes && !B_axes)
    return Status::OK();

  int i = int(a_size - 1);
  int j = int(b_size - 1);
  int k = ndim - 1;
  int a_idx = 0, b_idx = 0;

  for (; i >= 0 && j >= 0; --k) {
    auto A_dim = A_dims[i],
         B_dim = B_dims[j];

    if (A_dim != B_dim) {
      if (A_axes && A_dim == 1) {
        A_axes_data[a_idx] = gsl::narrow_cast<T>(k);
        a_idx++;
      }
      if (B_axes && B_dim == 1) {
        B_axes_data[b_idx] = (gsl::narrow_cast<T>(k));
        b_idx++;
      }
    }
    --i;
    --j;
  }

  if (i < 0) {
    if (A_axes) {
      for (; k >= 0; --k) {
        A_axes_data[a_idx] = gsl::narrow_cast<T>(k);
        a_idx++;
      }
    }
  } else {
    if (B_axes) {
      for (; k >= 0; --k) {
        B_axes_data[b_idx] = (gsl::narrow_cast<T>(k));
        b_idx++;
      }
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
