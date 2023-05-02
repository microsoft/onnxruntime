// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "batch_norm.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace std;
namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      BatchNormalization,                                          \
      kOnnxDomain,                                                 \
      7, 8,                                                        \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),  \
      BatchNorm<T>);                                               \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      BatchNormalization,                                          \
      kOnnxDomain,                                                 \
      9, 13,                                                       \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),  \
      BatchNorm<T>);                                               \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      BatchNormalization,                                          \
      kOnnxDomain,                                                 \
      14, 14,                                                      \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())   \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<T>()),  \
      BatchNorm<T>);                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      BatchNormalization,                                          \
      kOnnxDomain,                                                 \
      15,                                                          \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
      BatchNorm<T>);

template <typename T>
Status BatchNorm<T>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* B = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* mean = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* var = p_op_kernel_context->Input<Tensor>(4);

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, spatial_ == 1));

  const TensorShape& x_shape = X->Shape();
  const TensorShape& channel_shape = mean->Shape();

  Tensor* Y = p_op_kernel_context->Output(0, x_shape);
  Tensor* running_mean = p_op_kernel_context->Output(1, channel_shape);
  Tensor* running_var = p_op_kernel_context->Output(2, channel_shape);
  Tensor* saved_mean = p_op_kernel_context->Output(3, channel_shape);
  Tensor* saved_var = p_op_kernel_context->Output(4, channel_shape);

  auto x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->Data<T>());
  auto b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
  auto mean_data = reinterpret_cast<const CudaT*>(mean->Data<T>());
  auto var_data = reinterpret_cast<const CudaT*>(var->Data<T>());

  auto y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;

  CudnnTensor data_desc;
  vector<int64_t> new_dims;
  BatchNormHelper::NormalizeDims(x_shape, new_dims);
  ORT_RETURN_IF_ERROR(data_desc.Set(new_dims, CudnnTensor::GetDataType<CudaT>()));

  // For half data type, the alpha, beta, scale, B, mean, var need to be float type
  if (X->IsDataType<MLFloat16>()) {
    CudnnTensor scale_desc;
    ORT_RETURN_IF_ERROR(scale_desc.Set(new_dims, CudnnTensor::GetDataType<float>()));
    CudnnTensor bn_tensor_desc;
    ORT_RETURN_IF_ERROR(bn_tensor_desc.Set(data_desc, cudnn_batch_norm_mode_));

    // Convert the scale, B, mean, var to float
    const int64_t C = x_shape.GetDims()[1];
    auto f_scale = GetScratchBuffer<float>(C, p_op_kernel_context->GetComputeStream());
    auto f_B = GetScratchBuffer<float>(C, p_op_kernel_context->GetComputeStream());
    auto f_mean = GetScratchBuffer<float>(C, p_op_kernel_context->GetComputeStream());
    auto f_var = GetScratchBuffer<float>(C, p_op_kernel_context->GetComputeStream());
    Impl_Cast<CudaT, float>(Stream(p_op_kernel_context), scale_data, f_scale.get(), C);
    Impl_Cast<CudaT, float>(Stream(p_op_kernel_context), b_data, f_B.get(), C);
    Impl_Cast<CudaT, float>(Stream(p_op_kernel_context), mean_data, f_mean.get(), C);
    Impl_Cast<CudaT, float>(Stream(p_op_kernel_context), var_data, f_var.get(), C);

    CUDNN_RETURN_IF_ERROR(BatchNormalizationForwardInferenceHelper(
        GetCudnnHandle(p_op_kernel_context),
        cudnn_batch_norm_mode_,
        &alpha,
        &beta,
        data_desc,
        x_data,
        data_desc,
        y_data,
        bn_tensor_desc,
        f_scale.get(),
        f_B.get(),
        f_mean.get(),
        f_var.get(),
        epsilon_));

    return Status::OK();
  }

  CudnnTensor bn_tensor_desc;
  ORT_RETURN_IF_ERROR(bn_tensor_desc.Set(data_desc, cudnn_batch_norm_mode_));

  // in BatchNorm Forward Training mode if all 5 outputs present
  if (running_mean && running_var && saved_mean && saved_var) {
    auto running_mean_data = reinterpret_cast<CudaT*>(running_mean->MutableData<T>());
    auto running_var_data = reinterpret_cast<CudaT*>(running_var->MutableData<T>());
    auto saved_mean_data = reinterpret_cast<CudaT*>(saved_mean->MutableData<T>());
    auto saved_inv_var_data = reinterpret_cast<CudaT*>(saved_var->MutableData<T>());

    CUDNN_RETURN_IF_ERROR(BatchNormalizationForwardTrainingHelper(
        GetCudnnHandle(p_op_kernel_context),
        cudnn_batch_norm_mode_,
        &alpha,
        &beta,
        data_desc,
        x_data,
        data_desc,
        y_data,
        bn_tensor_desc,
        scale_data,
        b_data,
        momentum_,
        running_mean_data,
        running_var_data,
        epsilon_,
        saved_mean_data,
        saved_inv_var_data));
    // in BatchNorm Forward Inference mode if only Y output present
  } else {
    CUDNN_RETURN_IF_ERROR(BatchNormalizationForwardInferenceHelper(
        GetCudnnHandle(p_op_kernel_context),
        cudnn_batch_norm_mode_,
        &alpha,
        &beta,
        data_desc,
        x_data,
        data_desc,
        y_data,
        bn_tensor_desc,
        scale_data,
        b_data,
        mean_data,
        var_data,
        epsilon_));
  }
  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status BatchNorm<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
}  // namespace onnxruntime
