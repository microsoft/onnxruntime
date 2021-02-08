// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/rocm/math/softmax_grad.h"

#include "core/providers/common.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/math/softmax.h"
#include "core/providers/rocm/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace rocm {

template <typename T, bool is_log_softmax>
Status SoftMaxGradComputeHelper(
    hipStream_t stream,
    const T* dY,
    const TensorShape& input_shape,
    const T* Y,
    T* dX,
    miopenHandle_t handle,
    int64_t axis) {
  typedef typename ToHipType<T>::MappedType HipT;

  const int64_t normalized_axis = HandleNegativeAxis(axis, input_shape.NumDimensions());

  int64_t N = input_shape.SizeToDimension(normalized_axis);
  int64_t D = input_shape.SizeFromDimension(normalized_axis);
  std::vector<int64_t> dims({N, 1, 1, D});  // miopen expects 4D shape in NCHW format

  auto dY_data = reinterpret_cast<const HipT*>(dY);
  auto Y_data = reinterpret_cast<const HipT*>(Y);
  auto dX_data = reinterpret_cast<HipT*>(dX);

  if (D <= 1024 && D * sizeof(T) <= 4096) {
    dispatch_softmax_backward<HipT, HipT, AccumulationType_t<HipT>, is_log_softmax>(
        stream, dX_data, dY_data, Y_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N));
    return Status::OK();
  }

  const auto alpha = Consts<HipT>::One;
  const auto beta = Consts<HipT>::Zero;
  MiopenTensor input_tensor;
  MiopenTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(dims, MiopenTensor::GetDataType<HipT>()));
  ORT_RETURN_IF_ERROR(output_tensor.Set(dims, MiopenTensor::GetDataType<HipT>()));
  MIOPEN_RETURN_IF_ERROR(
      miopenSoftmaxBackward_V2(
          handle,
          &alpha,
          input_tensor,
          Y_data,
          input_tensor,
          dY_data,
          &beta,
          output_tensor,
          dX_data,
          is_log_softmax? MIOPEN_SOFTMAX_LOG : MIOPEN_SOFTMAX_ACCURATE,
          MIOPEN_SOFTMAX_MODE_INSTANCE));

  return Status::OK();
}

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      SoftmaxGrad,                                                              \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);                                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      LogSoftmaxGrad,                                                           \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
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
    return SoftMaxGradComputeHelper<T, true>(Stream(), dY_data, input_shape, Y_data, dX_data, MiopenHandle(), axis_);
  } else {
    return SoftMaxGradComputeHelper<T, false>(Stream(), dY_data, input_shape, Y_data, dX_data, MiopenHandle(), axis_);
  }
}

#define SPECIALIZED_GRADIENT(T)     \
  REGISTER_GRADIENT_KERNEL_TYPED(T) \
  template Status SoftmaxGrad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_GRADIENT(float)
// SPECIALIZED_GRADIENT(double)
SPECIALIZED_GRADIENT(MLFloat16)

}  // namespace rocm
}  // namespace onnxruntime
