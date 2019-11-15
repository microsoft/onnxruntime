// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/math/clip.h"
#include "core/providers/cuda/math/clip_impl.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      Clip,                                                                     \
      kOnnxDomain,                                                              \
      6, 10,                                                                    \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Clip_6<T>);                                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Clip,                                                                     \
      kOnnxDomain,                                                              \
      11,                                                                       \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Clip<T>);

template <typename T>
Status Clip_6<T>::ComputeInternal(OpKernelContext* ctx) const {
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

template <typename T>
Status Clip<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* min = ctx->Input<Tensor>(1);
  const auto* max = ctx->Input<Tensor>(2);

  auto min_val = -std::numeric_limits<T>::infinity();
  auto max_val = std::numeric_limits<T>::infinity();
  if (min) {
    ORT_ENFORCE(min->Shape().NumDimensions() == 0, "min should be a scalar.");
    CUDA_RETURN_IF_ERROR(cudaMemcpy(&min_val, min->template Data<T>(), sizeof(T), cudaMemcpyDeviceToHost));
  }
  if (max) {
    ORT_ENFORCE(max->Shape().NumDimensions() == 0, "max should be a scalar.");
    CUDA_RETURN_IF_ERROR(cudaMemcpy(&max_val, max->template Data<T>(), sizeof(T), cudaMemcpyDeviceToHost));
  }
  ORT_ENFORCE(min_val <= max_val);

  const Tensor& X = *ctx->Input<Tensor>(0);
  const TensorShape input_shape{X.Shape()};
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
