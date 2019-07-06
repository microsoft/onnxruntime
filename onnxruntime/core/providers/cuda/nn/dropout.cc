// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "dropout.h"

namespace onnxruntime {
namespace cuda {

Status CudnnDropoutState::Set(cudnnHandle_t handle, const TensorShape& shape, cudnnDataType_t type, float ratio) {
  if (!dropout_desc) {
    CUDNN_RETURN_IF_ERROR(cudnnCreateDropoutDescriptor(&dropout_desc));
  }
  if (!dropout_in_out_desc) {
    CUDNN_RETURN_IF_ERROR(cudnnCreateTensorDescriptor(&dropout_in_out_desc));
  }

  CUDNN_RETURN_IF_ERROR(cudnnSetTensor4dDescriptor(dropout_in_out_desc,
                                                   CUDNN_TENSOR_NCHW,  //TODO: is this always true?
                                                   type,
                                                   static_cast<int>(shape.Size()), 1, 1, 1));

  CUDNN_RETURN_IF_ERROR(cudnnDropoutGetStatesSize(handle, &dropout_state_size));
  CUDNN_RETURN_IF_ERROR(cudnnDropoutGetReserveSpaceSize(dropout_in_out_desc, &dropout_reserve_size));

  //Allocate memory for states and reserve space
  CUDA_RETURN_IF_ERROR(cudaMalloc(&states, dropout_state_size));

  //TODO: How is the seed in schema applied here
  {
    std::lock_guard<OrtMutex> lock(mutex);
    CUDNN_RETURN_IF_ERROR(cudnnSetDropoutDescriptor(dropout_desc, handle, ratio, states, dropout_state_size, /*seed*/ 0));
  }
  return Status::OK();
}

Status CudnnDropoutState::Release() {
  CUDNN_RETURN_IF_ERROR(cudnnDestroyTensorDescriptor(dropout_in_out_desc));
  CUDNN_RETURN_IF_ERROR(cudnnDestroyDropoutDescriptor(dropout_desc));
  CUDA_RETURN_IF_ERROR(cudaFree(states));
  return Status::OK();
}

#define REGISTER_KERNEL_TYPED(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      TrainableDropout,                                          \
      kOnnxDomain,                                               \
      9,                                                         \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .InputMemoryType<OrtMemTypeCPUInput>(1),               \
      TrainableDropout<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

template <typename T>
Status TrainableDropout<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const TensorShape& shape = X->Shape();
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());

  //Get Y_data
  auto Y = context->Output(0, shape);
  auto Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  //Get mask_data
  auto mask = context->Output(1, shape);
  auto mask_data = reinterpret_cast<CudaT*>(mask->template MutableData<bool>());

  // TODO(zuowei): fix dropout impl
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y_data, X_data, X->Size(), cudaMemcpyDeviceToDevice));
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(mask_data, 0, mask->Size()));

/*
  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->MutableInput<Tensor>(1);
  if (ratio) {
    ratio_data = *reinterpret_cast<float*>(ratio->template MutableData<T>());
    ORT_ENFORCE(ratio_data >= 0 && ratio_data < 1);
  }
  bool is_test = (ratio_data == 0);
  if (is_test) {
    if (Y_data != X_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y_data, X_data, X->Size(), cudaMemcpyDeviceToDevice));
    }
  } else {
    s_.Set(CudnnHandle(), shape, CudnnTensor::GetDataType<CudaT>(), ratio_data);
    //Start computing

    //TODO: Should it be mutex guarded? Pytorch doesn't, is it correct?
    CUDNN_RETURN_IF_ERROR(cudnnDropoutForward(
        CudnnHandle(),
        s_.dropout_desc,
        s_.dropout_in_out_desc,
        X_data,
        s_.dropout_in_out_desc,
        Y_data,
        mask_data,
        s_.dropout_reserve_size));

    s_.Release();
  }
  */
  return Status::OK();
}

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      TrainableDropoutGrad,                                      \
      kOnnxDomain,                                               \
      9,                                                         \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .InputMemoryType<OrtMemTypeCPUInput>(2),               \
      TrainableDropoutGrad<T>);

REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(float)
REGISTER_GRADIENT_KERNEL_TYPED(double)

template <typename T>
Status TrainableDropoutGrad<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* dY = context->Input<Tensor>(0);
  auto dY_data = reinterpret_cast<const CudaT*>(dY->template Data<T>());
  const TensorShape& shape = dY->Shape();

  Tensor* dX = context->Output(0, shape);
  auto dX_data = reinterpret_cast<CudaT*>(dX->template MutableData<T>());

  // TODO(zuowei): fix dropout impl
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dX_data, dY_data, dY->Size(), cudaMemcpyDeviceToDevice));

/*
  Tensor* mask = context->MutableInput<Tensor>(1);
  auto mask_data = reinterpret_cast<CudaT*>(mask->template MutableData<bool>());

  auto* ratio = context->MutableInput<Tensor>(2);
  float ratio_data = default_ratio_;
  if (ratio) {
    ratio_data = *reinterpret_cast<float*>(ratio->template MutableData<T>());
    ORT_ENFORCE(ratio_data >= 0 && ratio_data < 1);
  }

  s_.Set(CudnnHandle(), shape, CudnnTensor::GetDataType<CudaT>(), ratio_data);

  //Start computing

  //TODO: Should it be mutex guarded? Pytorch doesn't, is it correct?
  CUDNN_RETURN_IF_ERROR(cudnnDropoutBackward(
      CudnnHandle(),
      s_.dropout_desc,
      s_.dropout_in_out_desc,
      dY_data,
      s_.dropout_in_out_desc,
      dX_data,
      mask_data,
      s_.dropout_reserve_size));
  s_.Release();
  */
  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
