// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "softmax.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T, bool is_log_softmax>
Status SoftMaxComputeHelper(
    const T* X,
    const TensorShape& input_shape,
    T* Y,
    cudnnHandle_t handle,
    int64_t axis) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const int64_t normalized_axis = HandleNegativeAxis(axis, input_shape.NumDimensions());

  int64_t N = input_shape.SizeToDimension(normalized_axis);
  int64_t D = input_shape.SizeFromDimension(normalized_axis);
  auto Y_data = reinterpret_cast<CudaT*>(Y);
  auto X_data = reinterpret_cast<const CudaT*>(X);

  // cudnnSoftmaxForward/Backward is not optimal implementation.
  // TODO: remove cudnn path completely in the future.
  if (D == input_shape[normalized_axis] && D <= 1024 && D * sizeof(T) <= 4096) {
    dispatch_softmax_forward<CudaT, CudaT, AccType<T>, is_log_softmax>(Y_data, X_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N));
    return Status::OK();
  }

  std::vector<int64_t> dims({N, 1, 1, D});  // cudnn expects 4D shape in NCHW format

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(output_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>()));
  if (is_log_softmax) {
    CUDNN_RETURN_IF_ERROR(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_tensor, X_data, &beta, output_tensor, Y_data));
  } else {
    CUDNN_RETURN_IF_ERROR(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_tensor, X_data, &beta, output_tensor, Y_data));
  }

  return Status::OK();
}

#define SPECIALIZED_SOFTMAX_HELPER_IMPL(T)                                                                                            \
  template Status SoftMaxComputeHelper<T, false>(const T* input, const TensorShape& shape, T* Y, cudnnHandle_t handle, int64_t axis); \
  template Status SoftMaxComputeHelper<T, true>(const T* input, const TensorShape& shape, T* Y, cudnnHandle_t handle, int64_t axis);

SPECIALIZED_SOFTMAX_HELPER_IMPL(float)
SPECIALIZED_SOFTMAX_HELPER_IMPL(double)
SPECIALIZED_SOFTMAX_HELPER_IMPL(MLFloat16)

#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      Softmax,                                                                  \
      kOnnxDomain,                                                              \
      1, 10,                                                                    \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Softmax,                                                                  \
      kOnnxDomain,                                                              \
      11,                                                                       \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                              \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      LogSoftmax,                                                               \
      kOnnxDomain,                                                              \
      1, 10,                                                                    \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      LogSoftmax,                                                               \
      kOnnxDomain,                                                              \
      11,                                                                       \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);

template <typename T>
Status Softmax<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& input_shape{X->Shape()};
  const T* X_data = X->template Data<T>();
  T* Y_data = ctx->Output(0, input_shape)->template MutableData<T>();
  // special case when there is a dim value of 0 in the shape.
  if (input_shape.Size() == 0)
    return Status::OK();

  if (log_softmax_) {
    return SoftMaxComputeHelper<T, true>(X_data, input_shape, Y_data, CudnnHandle(), axis_);
  }
  else {
    return SoftMaxComputeHelper<T, false>(X_data, input_shape, Y_data, CudnnHandle(), axis_);
  }
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Softmax<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
}  // namespace onnxruntime
