// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/batch_norm_grad.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"

using namespace std;
namespace onnxruntime {
namespace cuda {

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      BatchNormalizationGrad,                                                              \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BatchNormalizationGrad<T>);

template <typename T>
Status BatchNormalizationGrad<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* dY = ctx->Input<Tensor>(0);
  const Tensor* X = ctx->Input<Tensor>(1);
  const Tensor* Scale = ctx->Input<Tensor>(2);
  const Tensor* saved_mean = ctx->Input<Tensor>(3);
  const Tensor* saved_variance = ctx->Input<Tensor>(4);
  const TensorShape input_shape = X->Shape();
  const TensorShape channel_shape = saved_mean->Shape();

  // no B here, but B has same size as Scale, so can validate inputs for gradient with this substitute
  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, Scale, Scale, saved_mean, saved_variance));

  auto dY_data = reinterpret_cast<const CudaT*>(dY->template Data<T>());
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto Scale_data = reinterpret_cast<const CudaT*>(Scale->template Data<T>());
  auto saved_mean_data = reinterpret_cast<const CudaT*>(saved_mean->template Data<T>());
  auto saved_variance_data = reinterpret_cast<const CudaT*>(saved_variance->template Data<T>());

  auto dX_data = reinterpret_cast<CudaT*>(ctx->Output(0, input_shape)->template MutableData<T>());
  auto dScale_data = reinterpret_cast<CudaT*>(ctx->Output(1, channel_shape)->template MutableData<T>());
  auto dBias_data = reinterpret_cast<CudaT*>(ctx->Output(2, channel_shape)->template MutableData<T>());

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;

  CudnnTensor input_tensor, scale_bias_tensor;
  vector<int64_t> new_dims;
  BatchNormHelper::NormalizeDims(input_shape, new_dims);
  ORT_RETURN_IF_ERROR(input_tensor.Set(new_dims, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(scale_bias_tensor.Set(input_tensor, cudnn_batch_norm_mode_));

  // note this is only valid for cudnnBatchNormalizationForwardTraining, not ForwardInference
  CUDNN_RETURN_IF_ERROR(
      cudnnBatchNormalizationBackward(
          CudnnHandle(),
          cudnn_batch_norm_mode_,
          &alpha,
          &beta,
          &alpha,
          &beta,
          input_tensor,
          X_data,
          input_tensor,
          dY_data,
          input_tensor,
          dX_data,
          scale_bias_tensor,
          Scale_data,
          dScale_data,
          dBias_data,
          epsilon_,
          saved_mean_data,
          saved_variance_data));
  return Status::OK();
}

#define SPECIALIZED_GRADIENT(T)     \
  REGISTER_GRADIENT_KERNEL_TYPED(T) \
  template Status BatchNormalizationGrad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_GRADIENT(float)
SPECIALIZED_GRADIENT(double)

}  // namespace cuda
}  // namespace onnxruntime
