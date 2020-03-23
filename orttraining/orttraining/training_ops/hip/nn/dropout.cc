// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"

#include "orttraining/training_ops/hip/nn/dropout.h"

namespace onnxruntime {
namespace hip {

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      TrainableDropout,                                               \
      kOnnxDomain,                                                    \
      9,                                                              \
      T,                                                              \
      kHipExecutionProvider,                                         \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())  \
          .InputMemoryType<OrtMemTypeCPUInput>(1),                    \
      TrainableDropout<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

template <typename T>
Status TrainableDropout<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToHipType<T>::MappedType HipT;

  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& shape = X->Shape();
  auto X_data = reinterpret_cast<const HipT*>(X->template Data<T>());
  const int64_t N = shape.Size();

  //Get Y_data
  auto Y = context->Output(0, shape);
  auto Y_data = reinterpret_cast<HipT*>(Y->template MutableData<T>());

  //Get mask_data
  auto mask = context->Output(1, shape);
  ORT_ENFORCE(!mask || mask->Shape().Size() == N);
  IAllocatorUniquePtr<bool> temp_mask_buffer{};  // buffer to use if mask is not provided
  bool* const mask_data = [this, N, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableData<bool>();
    temp_mask_buffer = GetScratchBuffer<bool>(N);
    return temp_mask_buffer.get();
  }();

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(1);
  if (ratio) {
    ratio_data = *(ratio->template Data<float>());
  }
  ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f);

  DropoutKernelImpl(GetDeviceProp(), N, ratio_data, generator_, X_data, Y_data, mask_data);

  return Status::OK();
}

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      TrainableDropoutGrad,                                           \
      kMSDomain,                                                    \
      1,                                                              \
      T,                                                              \
      kHipExecutionProvider,                                         \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())  \
          .InputMemoryType<OrtMemTypeCPUInput>(2),                    \
      TrainableDropoutGrad<T>);

REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(float)
REGISTER_GRADIENT_KERNEL_TYPED(double)

template <typename T>
Status TrainableDropoutGrad<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToHipType<T>::MappedType HipT;

  auto dY = context->Input<Tensor>(0);
  const TensorShape& shape = dY->Shape();
  auto dY_data = reinterpret_cast<const HipT*>(dY->template Data<T>());
  const int64_t N = shape.Size();

  auto mask = context->Input<Tensor>(1);
  ORT_ENFORCE(mask->Shape().Size() == N);

  auto dX = context->Output(0, shape);
  auto dX_data = reinterpret_cast<HipT*>(dX->template MutableData<T>());

  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(2);
  if (ratio) {
    ratio_data = *(ratio->template Data<float>());
  }
  ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f);

  const bool* mask_data = mask->template Data<bool>();
  DropoutGradientKernelImpl(N, dY_data, mask_data, ratio_data, dX_data);

  return Status::OK();
}
}  // namespace hip
}  // namespace onnxruntime
