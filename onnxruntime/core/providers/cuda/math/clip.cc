// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/math/clip.h"
#include "core/providers/cuda/math/clip_impl.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Clip,                                                       \
      kOnnxDomain,                                                \
      6,                                                          \
      10,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Clip<T>);                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Clip,                                                       \
      kOnnxDomain,                                                \
      11,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                 \
          .InputMemoryType<OrtMemTypeCPUInput>(2)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Clip<T>);

template <typename T>
Status Clip<T>::ComputeInternal(OpKernelContext* ctx) const {
  T min_val = min_;
  T max_val = max_;
  if (is_min_max_input_) {
    const auto* min_input = ctx->Input<Tensor>(1);
    const auto* max_input = ctx->Input<Tensor>(2);
    if (min_input) {
      ORT_ENFORCE(min_input->Shape().NumDimensions() == 0, "min should be a scalar.");
      min_val = *(min_input->template Data<T>());
    }
    if (max_input) {
      ORT_ENFORCE(max_input->Shape().NumDimensions() == 0, "max should be a scalar.");
      max_val = *(max_input->template Data<T>());
    }
    ORT_ENFORCE(min_val <= max_val);
  }

  const Tensor& X = *ctx->Input<Tensor>(0);
  const TensorShape& input_shape{X.Shape()};
  Tensor* Y = ctx->Output(0, input_shape);
  size_t count = input_shape.Size();
  if (count > 0) {
    auto* y_data = Y->template MutableData<T>();
    const auto* x_data = X.template Data<T>();
    ClipImpl<T>(x_data, y_data, min_val, max_val, count);
  }

  return Status::OK();
}

REGISTER_KERNEL_TYPED(float)

}  // namespace cuda
}  // namespace onnxruntime
