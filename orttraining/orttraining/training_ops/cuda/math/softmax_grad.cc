// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/softmax_grad.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      SoftmaxGrad,                                                              \
      kOnnxDomain,                                                              \
      9,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);

template <typename T>
Status SoftmaxGrad<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* dY = ctx->Input<Tensor>(0);
  const TensorShape input_shape{dY->Shape()};

  const Tensor* Y = ctx->Input<Tensor>(1);

  const int64_t normalized_axis = HandleNegativeAxis(axis_, input_shape.NumDimensions());

  int64_t N = input_shape.SizeToDimension(normalized_axis);
  int64_t D = input_shape.SizeFromDimension(normalized_axis);
  std::vector<int64_t> dims({N, 1, 1, D});  // cudnn expects 4D shape in NCHW format

  auto dY_data = reinterpret_cast<const CudaT*>(dY->template Data<T>());
  auto Y_data = reinterpret_cast<const CudaT*>(Y->template Data<T>());
  auto dX_data = reinterpret_cast<CudaT*>(ctx->Output(0, input_shape)->template MutableData<T>());

  if (D == input_shape[normalized_axis] && D <= 1024 && D * sizeof(T) <= 4096) {
    dispatch_softmax_backward<CudaT, CudaT, AccType<T>, false>(dX_data, dY_data, Y_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N));
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
          CudnnHandle(),
          CUDNN_SOFTMAX_ACCURATE,
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

#define SPECIALIZED_GRADIENT(T)     \
  REGISTER_GRADIENT_KERNEL_TYPED(T) \
  template Status SoftmaxGrad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_GRADIENT(float)
SPECIALIZED_GRADIENT(double)
SPECIALIZED_GRADIENT(MLFloat16)

}  // namespace cuda
}  // namespace onnxruntime
