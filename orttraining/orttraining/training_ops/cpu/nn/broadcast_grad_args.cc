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
  const Tensor* a_shape = context->Input<Tensor>(0);
  const Tensor* b_shape = context->Input<Tensor>(1);
  const T* A_dims = a_shape->template Data<T>();
  const T* B_dims = b_shape->template Data<T>();

  const T a_size = a_shape->Shape().Size();
  const T b_size = b_shape->Shape().Size();

  T ndim = std::max(a_size, b_size);
  std::vector<T> a_axes, b_axes;

  T i = a_size - 1;
  T j = b_size - 1;
  T k = ndim - 1;

  for (; i >= 0 && j >= 0; --k) {
    auto A_dim = A_dims[i],
         B_dim = B_dims[j];

    if (A_dim != B_dim) {
      if (A_dim == 1) {
        a_axes.push_back(k);
      } else if (B_dim == 1) {
        b_axes.push_back(k);
      } else {
        TensorShape a(A_dims, a_size);
        TensorShape b(B_dims, b_size);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Broadcast is not possible between inputs of shapes: ",
                               a, " and ", b);
      }
    }
    --i;
    --j;
  }

  if (i < 0) {
    for (; k >= 0; --k) {
      a_axes.push_back(k);
    }

  } else {
    for (; k >= 0; --k) {
      b_axes.push_back(k);
    }
  }

  Tensor* A_axes = context->Output(0, {static_cast<T>(a_axes.size())});
  T* A_axes_data = A_axes->template MutableData<T>();
  std::copy(a_axes.begin(), a_axes.end(), A_axes_data);
  Tensor* B_axes = context->Output(1, {static_cast<T>(b_axes.size())});
  T* B_axes_data = B_axes->template MutableData<T>();
  std::copy(b_axes.begin(), b_axes.end(), B_axes_data);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
