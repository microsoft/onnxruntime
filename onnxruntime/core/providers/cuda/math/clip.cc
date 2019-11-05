// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/math/clip.h"
#include "core/providers/cuda/math/clip_impl.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Clip,                                                                     \
      kOnnxDomain,                                                              \
      6,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Clip<T>);

template <typename T>
Status Clip<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& X = *ctx->Input<Tensor>(0);
  const TensorShape input_shape{X.Shape()};
  Tensor* Y = ctx->Output(0, input_shape);

  size_t count = input_shape.Size();

  if (count > 0) {
    auto* y_data = Y->template MutableData<T>();
    const auto* x_data = X.template Data<T>();
    ClipImpl<T>(x_data, y_data, min_, max_, count);
  }

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Clip<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)

}  // namespace cuda
}  // namespace onnxruntime
