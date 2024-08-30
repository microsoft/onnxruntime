// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/nn/broadcast_grad_args.h"

#include "core/common/narrow.h"

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
  const auto A_dims = a_shape->template DataAsSpan<T>();
  const auto B_dims = b_shape->template DataAsSpan<T>();

  const T a_size = a_shape->Shape().Size();
  const T b_size = b_shape->Shape().Size();

  T ndim = std::max(a_size, b_size);
  std::vector<T> a_axes, b_axes;

  T i = a_size - 1;
  T j = b_size - 1;
  T k = ndim - 1;

  for (; i >= 0 && j >= 0; --k) {
    auto A_dim = A_dims[narrow<size_t>(i)],
         B_dim = B_dims[narrow<size_t>(j)];

    if (A_dim != B_dim) {
      if (A_dim == 1) {
        a_axes.push_back(k);
      } else if (B_dim == 1) {
        b_axes.push_back(k);
      } else {
        TensorShape a(A_dims);
        TensorShape b(B_dims);
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
  if (A_axes) {  // verify as A_axes is an optional output
    T* A_axes_data = A_axes->template MutableData<T>();
    std::copy(a_axes.begin(), a_axes.end(), A_axes_data);
  }

  Tensor* B_axes = context->Output(1, {static_cast<T>(b_axes.size())});
  if (B_axes) {  // verify as B_axes is an optional output
    T* B_axes_data = B_axes->template MutableData<T>();
    std::copy(b_axes.begin(), b_axes.end(), B_axes_data);
  }
  if (!A_axes && !B_axes) {
    LOGS_DEFAULT(WARNING) << "No output found for op BroadcastGradientArgs.";
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
