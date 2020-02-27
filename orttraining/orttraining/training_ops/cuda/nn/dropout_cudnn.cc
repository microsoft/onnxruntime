// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "orttraining/training_ops/cuda/nn/dropout_cudnn.h"

namespace onnxruntime {
namespace cuda {
DropoutBase::CudnnDropoutState::CudnnDropoutState(cudnnHandle_t handle) : ratio_(0.0f) {
  CUDNN_CALL_THROW(cudnnCreateDropoutDescriptor(&dropout_desc));
  CUDNN_CALL_THROW(cudnnCreateTensorDescriptor(&dropout_in_out_desc));
  CUDNN_CALL_THROW(cudnnDropoutGetStatesSize(handle, &dropout_state_size));
  //Allocate memory for states and reserve space
  CUDA_CALL_THROW(cudaMalloc(&states, dropout_state_size));
}

Status DropoutBase::CudnnDropoutState::Set(cudnnHandle_t handle, const TensorShape& shape, cudnnDataType_t type, float ratio) {
  CUDNN_RETURN_IF_ERROR(cudnnSetTensor4dDescriptor(dropout_in_out_desc,
                                                   CUDNN_TENSOR_NCHW,  //TODO: is this always true?
                                                   type,
                                                   static_cast<int>(shape.Size()), 1, 1, 1));
  CUDNN_RETURN_IF_ERROR(cudnnDropoutGetReserveSpaceSize(dropout_in_out_desc, &dropout_reserve_size));

  if (ratio != ratio_) {
    ratio_ = ratio;
    {
      std::lock_guard<OrtMutex> lock(mutex);
      //TODO: How is the seed in schema applied here
      CUDNN_RETURN_IF_ERROR(cudnnSetDropoutDescriptor(dropout_desc, handle, ratio, states, dropout_state_size, /*seed*/ 0));
    }
  }
  return Status::OK();
}

DropoutBase::CudnnDropoutState::~CudnnDropoutState() {
  CUDNN_CALL_THROW(cudnnDestroyTensorDescriptor(dropout_in_out_desc));
  CUDNN_CALL_THROW(cudnnDestroyDropoutDescriptor(dropout_desc));
  CUDA_CALL_THROW(cudaFree(states));
}

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      Dropout,                                                        \
      kOnnxDomain,                                                    \
      12,                                                             \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())  \
          .InputMemoryType<OrtMemTypeCPUInput>(1),                    \
      DropoutCudnn<T>);

// REGISTER_KERNEL_TYPED(MLFloat16)
// REGISTER_KERNEL_TYPED(float)
// REGISTER_KERNEL_TYPED(double)

template <typename T>
Status DropoutCudnn<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const TensorShape& shape = X->Shape();
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());

  //Get Y_data
  auto Y = context->Output(0, shape);
  auto Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y_data, X_data, X->SizeInBytes(), cudaMemcpyDeviceToDevice));
  //Get mask_data
  auto mask = context->Output(1, shape);
  CudaT* mask_data = nullptr;
  if (mask) {
    mask_data = reinterpret_cast<CudaT*>(mask->template MutableData<bool>());
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(mask_data, 0, mask->SizeInBytes()));
  }

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(1);
  if (ratio) {
    ratio_data = *(ratio->template Data<float>());
    ORT_ENFORCE(ratio_data >= 0 && ratio_data < 1);
  }
  bool is_test = (ratio_data == 0);
  if (is_test) {
    if (Y_data != X_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y_data, X_data, X->SizeInBytes(), cudaMemcpyDeviceToDevice));
    }
  } else {
    ORT_RETURN_IF_ERROR(s_.Set(CudnnHandle(), shape, CudnnTensor::GetDataType<CudaT>(), ratio_data));

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
  }
  return Status::OK();
}

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      DropoutGrad,                                                    \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())  \
          .InputMemoryType<OrtMemTypeCPUInput>(2),                    \
      DropoutCudnnGrad<T>);

// REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16)
// REGISTER_GRADIENT_KERNEL_TYPED(float)
// REGISTER_GRADIENT_KERNEL_TYPED(double)

template <typename T>
Status DropoutCudnnGrad<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  auto dY = context->Input<Tensor>(0);
  const TensorShape& shape = dY->Shape();
  auto dY_data = reinterpret_cast<const CudaT*>(dY->template Data<T>());

  auto mask = context->Input<Tensor>(1);

  auto dX = context->Output(0, shape);
  auto dX_data = reinterpret_cast<CudaT*>(dX->template MutableData<T>());

  auto ratio = context->Input<Tensor>(2);
  float ratio_data = default_ratio_;

  if (ratio) {
    ratio_data = *(ratio->template Data<float>());
    ORT_ENFORCE(ratio_data >= 0 && ratio_data < 1);
  }

  bool is_test = (ratio_data == 0);
  if (is_test) {
    if (dX_data != dY_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dX_data, dY_data, dX->SizeInBytes(), cudaMemcpyDeviceToDevice));
    }
  } else {
    ORT_RETURN_IF_ERROR(s_.Set(CudnnHandle(), shape, CudnnTensor::GetDataType<CudaT>(), ratio_data));

    //Start computing
    auto mask_data = reinterpret_cast<CudaT*>(const_cast<bool*>(mask->template Data<bool>()));

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
  }
  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
