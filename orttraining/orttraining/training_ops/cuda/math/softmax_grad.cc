// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/softmax_grad.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T, bool is_log_softmax>
Status SoftMaxGradComputeHelper(
    const T* dY,
    const TensorShape& input_shape,
    const T* Y,
    T* dX,
    cudnnHandle_t handle,
    int64_t axis) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const int64_t normalized_axis = HandleNegativeAxis(axis, input_shape.NumDimensions());

  int64_t N = input_shape.SizeToDimension(normalized_axis);
  int64_t D = input_shape.SizeFromDimension(normalized_axis);
  std::vector<int64_t> dims({N, 1, 1, D});  // cudnn expects 4D shape in NCHW format

  auto dY_data = reinterpret_cast<const CudaT*>(dY);
  auto Y_data = reinterpret_cast<const CudaT*>(Y);
  auto dX_data = reinterpret_cast<CudaT*>(dX);

  if (D == input_shape[normalized_axis] && D <= 1024 && D * sizeof(T) <= 4096) {
    dispatch_softmax_backward<CudaT, CudaT, AccType<T>, is_log_softmax>(dX_data, dY_data, Y_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N));
    return Status::OK();
  }

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(output_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>()));
  CUDNN_RETURN_IF_ERROR(
      cudnnSoftmaxBackward(
          handle,
          is_log_softmax? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE,
          CUDNN_SOFTMAX_MODE_INSTANCE,
          &alpha,
          input_tensor,
          Y_data,
          input_tensor,
          dY_data,
          &beta,
          output_tensor,
          dX_data));

  return Status::OK();
}


#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      SoftmaxGrad,                                                              \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);                                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      LogSoftmaxGrad,                                                           \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);

template <typename T>
Status SoftmaxGrad<T>::ComputeInternal(OpKernelContext* ctx) const {

  const Tensor* dY = ctx->Input<Tensor>(0);
  const TensorShape& input_shape{dY->Shape()};
  const Tensor* Y = ctx->Input<Tensor>(1);
  Tensor* dX = ctx->Output(0, input_shape);

  const T* dY_data = dY->template Data<T>();
  const T* Y_data = Y->template Data<T>();
  T* dX_data = dX->template MutableData<T>();

  if (log_softmax_) {
    return SoftMaxGradComputeHelper<T, true>(dY_data, input_shape, Y_data, dX_data, CudnnHandle(), axis_);
  }
  else {
    return SoftMaxGradComputeHelper<T, false>(dY_data, input_shape, Y_data, dX_data, CudnnHandle(), axis_);
  }
}

#define SPECIALIZED_GRADIENT(T)     \
  REGISTER_GRADIENT_KERNEL_TYPED(T) \
  template Status SoftmaxGrad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_GRADIENT(float)
SPECIALIZED_GRADIENT(double)
SPECIALIZED_GRADIENT(MLFloat16)

}  // namespace cuda
}  // namespace onnxruntime
