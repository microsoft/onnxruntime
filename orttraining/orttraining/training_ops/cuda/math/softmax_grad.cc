// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/softmax_grad.h"

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {

template <typename T, bool is_log_softmax>
Status SoftMaxGradComputeHelper(
    cudaStream_t stream,
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

  if (D <= 1024 && D * sizeof(T) <= 4096) {
    dispatch_softmax_backward<CudaT, CudaT, AccumulationType_t<CudaT>, is_log_softmax>(
        stream, dX_data, dY_data, Y_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N));
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
          is_log_softmax ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE,
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

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
// cudnnSoftmaxForward/Backward doesn't support BFloat16.
#define SPECIALIZED_SOFTMAXGRAD_HELPER_IMPL_BFloat16(is_log_softmax)                                                     \
  template <>                                                                                                            \
  Status SoftMaxGradComputeHelper<BFloat16, is_log_softmax>(                                                             \
      cudaStream_t stream,                                                                                               \
      const BFloat16* dY,                                                                                                \
      const TensorShape& input_shape,                                                                                    \
      const BFloat16* Y,                                                                                                 \
      BFloat16* dX,                                                                                                      \
      cudnnHandle_t,                                                                                                     \
      int64_t axis) {                                                                                                    \
    typedef typename ToCudaType<BFloat16>::MappedType CudaT;                                                             \
    const int64_t normalized_axis = HandleNegativeAxis(axis, input_shape.NumDimensions());                               \
    int64_t N = input_shape.SizeToDimension(normalized_axis);                                                            \
    int64_t D = input_shape.SizeFromDimension(normalized_axis);                                                          \
    auto dY_data = reinterpret_cast<const CudaT*>(dY);                                                                   \
    auto Y_data = reinterpret_cast<const CudaT*>(Y);                                                                     \
    auto dX_data = reinterpret_cast<CudaT*>(dX);                                                                         \
    dispatch_softmax_backward<CudaT, CudaT, AccumulationType_t<CudaT>, is_log_softmax>(                                  \
        stream, dX_data, dY_data, Y_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N)); \
    return Status::OK();                                                                                                 \
  }

SPECIALIZED_SOFTMAXGRAD_HELPER_IMPL_BFloat16(true)
    SPECIALIZED_SOFTMAXGRAD_HELPER_IMPL_BFloat16(false)
#endif

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      SoftmaxGrad,                                                                         \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);                                                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      LogSoftmaxGrad,                                                                      \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
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
    return SoftMaxGradComputeHelper<T, true>(Stream(), dY_data, input_shape, Y_data, dX_data, CudnnHandle(), axis_);
  } else {
    return SoftMaxGradComputeHelper<T, false>(Stream(), dY_data, input_shape, Y_data, dX_data, CudnnHandle(), axis_);
  }
}

#define SPECIALIZED_GRADIENT(T)     \
  REGISTER_GRADIENT_KERNEL_TYPED(T) \
  template Status SoftmaxGrad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_GRADIENT(float)
SPECIALIZED_GRADIENT(double)
SPECIALIZED_GRADIENT(MLFloat16)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
SPECIALIZED_GRADIENT(BFloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime
