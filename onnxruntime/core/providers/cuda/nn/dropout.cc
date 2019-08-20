// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dropout.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      TrainableDropout,                                               \
      kOnnxDomain,                                                    \
      9,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
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
  typedef typename ToCudaType<T>::MappedType CudaT;

  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& shape = X->Shape();
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());

  //Get Y_data
  auto Y = context->Output(0, shape);
  auto Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  //Get mask_data
  auto mask = context->Output(1, shape);

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(1);
  if (ratio) {
    ratio_data = *(ratio->template Data<float>());
  }
  ORT_ENFORCE(ratio_data >= 0 && ratio_data < 1);

  bool is_test = (ratio_data == 0);
  if (is_test) {
    if (Y_data != X_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y_data, X_data, X->Size(), cudaMemcpyDeviceToDevice));
    }
  } else {
    const int64_t N = shape.Size();
    ORT_ENFORCE(mask->Shape().Size() == N);
    bool* mask_data = mask->template MutableData<bool>();
    DropoutKernelImpl(N, ratio_data, generator_, X_data, Y_data, mask_data);
  }

  return Status::OK();
}

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      TrainableDropoutGrad,                                           \
      kOnnxDomain,                                                    \
      9,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
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
  typedef typename ToCudaType<T>::MappedType CudaT;

  auto dY = context->Input<Tensor>(0);
  const TensorShape& shape = dY->Shape();
  auto dY_data = reinterpret_cast<const CudaT*>(dY->template Data<T>());

  auto mask = context->Input<Tensor>(1);

  auto dX = context->Output(0, shape);
  auto dX_data = reinterpret_cast<CudaT*>(dX->template MutableData<T>());

  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(2);
  if (ratio) {
    ratio_data = *(ratio->template Data<float>());
  }
  ORT_ENFORCE(ratio_data >= 0 && ratio_data < 1);

  bool is_test = (ratio_data == 0);
  if (is_test) {
    if (dX_data != dY_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dX_data, dY_data, dX->Size(), cudaMemcpyDeviceToDevice));
    }
  } else {
    const float scale = 1.f / (1.f - ratio_data);
    const int64_t N = shape.Size();
    ORT_ENFORCE(mask->Shape().Size() == N);
    const bool* mask_data = mask->template Data<bool>();
    DropoutGradientKernelImpl(N, dY_data, mask_data, scale, dX_data);
  }

  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
