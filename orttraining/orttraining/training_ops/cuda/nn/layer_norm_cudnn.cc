// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "orttraining/training_ops/cuda/nn/layer_norm_cudnn.h"
#include "orttraining/training_ops/cuda/nn/layer_norm_impl_cudnn.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LayerNormalization,                                         \
      kOnnxDomain,                                                \
      9,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LayerNormCudnn<T>);

//REGISTER_KERNEL_TYPED(float)
//REGISTER_KERNEL_TYPED(double)
//REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
LayerNormCudnn<T>::LayerNormCudnn(const OpKernelInfo& op_kernel_info)
    : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = ClampCudnnBatchNormEpsilon(tmp_epsilon);
}

template <typename T>
Status LayerNormCudnn<T>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  // Inputs
  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* bias = p_op_kernel_context->Input<Tensor>(2);

  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->template Data<T>());
  auto bias_data = reinterpret_cast<const CudaT*>(bias->template Data<T>());

  const TensorShape& x_shape = X->Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());

  auto N = x_shape.SizeToDimension(axis);
  auto M = x_shape.SizeFromDimension(axis);
  ORT_ENFORCE(M != 1, "M should not be 1");
  // Outputs
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);
  std::vector<int64_t> mean_inv_std_var_dim;
  mean_inv_std_var_dim.reserve(x_shape.NumDimensions());
  for (int i = 0; i < static_cast<int>(x_shape.NumDimensions()); ++i) {
    if (i < axis) {
      mean_inv_std_var_dim.emplace_back(x_shape.GetDims()[i]);
    } else {
      mean_inv_std_var_dim.emplace_back(1);
    }
  }
  Tensor* mean = p_op_kernel_context->Output(1, TensorShape(mean_inv_std_var_dim));
  Tensor* inv_std_var = p_op_kernel_context->Output(2, TensorShape(mean_inv_std_var_dim));

  auto Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
  auto mean_data = reinterpret_cast<CudaT*>(mean->template MutableData<T>());
  auto inv_std_var_data = reinterpret_cast<CudaT*>(inv_std_var->template MutableData<T>());

  CudnnTensor data_desc;
  ORT_RETURN_IF_ERROR(data_desc.Set({1, N, M, 1}, CudnnTensor::GetDataType<CudaT>()));

  CudnnTensor scale_bias_desc;
  ORT_RETURN_IF_ERROR(scale_bias_desc.Set({1, N, 1, 1}, CudnnTensor::GetDataType<CudaT>()));

  auto one_scale = GetScratchBuffer<CudaT>(N);
  Fill<CudaT>(one_scale.get(), one, N);
  auto zero_bias = GetScratchBuffer<CudaT>(N);
  CUDA_RETURN_IF_ERROR(cudaMemset(zero_bias.get(), 0, N * sizeof(CudaT)));

  // should care running_mean and running_variance?
  //auto running_mean = GetScratchBuffer<CudaT>(N);
  //auto running_variance = GetScratchBuffer<CudaT>(N);
  // first, compute mean and variance using cudnnBatchNorm training
  CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardTraining(
      CudnnHandle(),
      CUDNN_BATCHNORM_SPATIAL,
      &one,
      &zero,
      data_desc,
      X_data,
      data_desc,
      Y_data,
      scale_bias_desc,
      one_scale.get(),
      zero_bias.get(),
      1.0f,
      nullptr,  //running_mean.get(),
      nullptr,  //running_variance.get(),
      epsilon_,
      mean_data,
      inv_std_var_data));

  LayerNormLinearKernel<CudaT>(N, M, X_data, scale_data, bias_data, Y_data);

  return Status::OK();
}

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LayerNormalizationGrad,                                     \
      kOnnxDomain,                                                \
      9,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LayerNormCudnnGrad<T>);

//REGISTER_GRADIENT_KERNEL_TYPED(float)
//REGISTER_GRADIENT_KERNEL_TYPED(double)
//REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16)

template <typename T>
LayerNormCudnnGrad<T>::LayerNormCudnnGrad(const OpKernelInfo& op_kernel_info)
    : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());

  typedef typename ToCudaType<T>::MappedType CudaT;
  if (!reduce_sum_desc_) {
    CUDNN_CALL_THROW(cudnnCreateReduceTensorDescriptor(&reduce_sum_desc_));
  }
  CUDNN_CALL_THROW(cudnnSetReduceTensorDescriptor(
      reduce_sum_desc_,
      CUDNN_REDUCE_TENSOR_ADD,
      CudnnTensor::GetDataType<CudaT>(),
      CUDNN_PROPAGATE_NAN,
      CUDNN_REDUCE_TENSOR_NO_INDICES,
      CUDNN_32BIT_INDICES));

  if (!reduce_mean_desc_) {
    CUDNN_CALL_THROW(cudnnCreateReduceTensorDescriptor(&reduce_mean_desc_));
  }
  CUDNN_CALL_THROW(cudnnSetReduceTensorDescriptor(
      reduce_mean_desc_,
      CUDNN_REDUCE_TENSOR_AVG,
      CudnnTensor::GetDataType<CudaT>(),
      CUDNN_PROPAGATE_NAN,
      CUDNN_REDUCE_TENSOR_NO_INDICES,
      CUDNN_32BIT_INDICES));
}

template <typename T>
Status LayerNormCudnnGrad<T>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  // Inputs
  const Tensor* Y_grad = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* X = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* mean = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* inv_std_var = p_op_kernel_context->Input<Tensor>(4);

  auto Y_grad_data = reinterpret_cast<const CudaT*>(Y_grad->template Data<T>());
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->template Data<T>());
  auto mean_data = reinterpret_cast<const CudaT*>(mean->template Data<T>());
  auto inv_std_var_data = reinterpret_cast<const CudaT*>(inv_std_var->template Data<T>());

  const TensorShape& x_shape = X->Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
  auto N = x_shape.SizeToDimension(axis);
  auto M = x_shape.SizeFromDimension(axis);
  ORT_ENFORCE(M != 1, "M should not be 1");
  // Outputs
  Tensor* X_grad = p_op_kernel_context->Output(0, x_shape);
  auto X_grad_data = reinterpret_cast<CudaT*>(X_grad->template MutableData<T>());

  Tensor* scale_grad = p_op_kernel_context->Output(1, scale->Shape());
  Tensor* bias_grad = p_op_kernel_context->Output(2, scale->Shape());
  auto scale_grad_data = reinterpret_cast<CudaT*>(scale_grad->template MutableData<T>());
  auto bias_grad_data = reinterpret_cast<CudaT*>(bias_grad->template MutableData<T>());

  // Calculating gradients.
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set({1, N, M, 1}, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(output_tensor.Set({1, N, 1, 1}, CudnnTensor::GetDataType<CudaT>()));

  size_t workspace_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(CudnnHandle(), reduce_mean_desc_, input_tensor, output_tensor, &workspace_bytes));
  auto workspace_cuda = GetScratchBuffer<CudaT>(workspace_bytes);

  size_t indices_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(CudnnHandle(), reduce_mean_desc_, input_tensor, output_tensor, &indices_bytes));
  auto indices_cuda = GetScratchBuffer<uint32_t>(indices_bytes);

  auto A = GetScratchBuffer<CudaT>(N * M);
  auto B = GetScratchBuffer<CudaT>(N * M);
  auto C = GetScratchBuffer<CudaT>(N * M);
  // A, B, C are calculated as below:
  // A = Y_grad * (X - mean(X)) * inv_std_var
  // B = Y_grad * scale * inv_std_var
  // c = Y_grad * scale  * inv_std_var * (X - mean)  * inv_std_var
  LayerNormGradInternalKernel(N, M, Y_grad_data, X_data, mean_data, inv_std_var_data, scale_data, A.get(), B.get(), C.get());

  // mean_B = mean(Y_grad * scale * inv_std_var)
  auto mean_B = GetScratchBuffer<CudaT>(N);
  CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
      CudnnHandle(),
      reduce_mean_desc_,
      indices_cuda.get(),
      indices_bytes,
      workspace_cuda.get(),
      workspace_bytes,
      &one,
      input_tensor,
      B.get(),
      &zero,
      output_tensor,
      mean_B.get()));

  // mean_C = mean(Y_grad * scale  * inv_std_var * (X - mean)  * inv_std_var)
  auto mean_C = GetScratchBuffer<CudaT>(N);
  CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
      CudnnHandle(),
      reduce_mean_desc_,
      indices_cuda.get(),
      indices_bytes,
      workspace_cuda.get(),
      workspace_bytes,
      &one,
      input_tensor,
      C.get(),
      &zero,
      output_tensor,
      mean_C.get()));

  // X_grad =  Y_grad * scale *inv_std_var - mean_B - (X - mean) * inv_std_var * mean_C
  LayerNormGradXKernel(N, M, X_data, mean_data, B.get(), mean_B.get(), mean_C.get(), inv_std_var_data, X_grad_data);

  // bias_grad = sum(Y_grad)
  ORT_RETURN_IF_ERROR(input_tensor.Set({1, N, M, 1}, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(output_tensor.Set({1, 1, M, 1}, CudnnTensor::GetDataType<CudaT>()));
  workspace_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(CudnnHandle(), reduce_sum_desc_, input_tensor, output_tensor, &workspace_bytes));
  workspace_cuda = GetScratchBuffer<CudaT>(workspace_bytes);

  indices_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(CudnnHandle(), reduce_sum_desc_, input_tensor, output_tensor, &indices_bytes));
  indices_cuda = GetScratchBuffer<uint32_t>(indices_bytes);
  CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
      CudnnHandle(),
      reduce_sum_desc_,
      indices_cuda.get(),
      indices_bytes,
      workspace_cuda.get(),
      workspace_bytes,
      &one,
      input_tensor,
      Y_grad_data,
      &zero,
      output_tensor,
      bias_grad_data));

  // scale_grad = sum(Y_grad * (X - mean(X)) * inv_std_var)
  CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
      CudnnHandle(),
      reduce_sum_desc_,
      indices_cuda.get(),
      indices_bytes,
      workspace_cuda.get(),
      workspace_bytes,
      &one,
      input_tensor,
      A.get(),
      &zero,
      output_tensor,
      scale_grad_data));

  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
