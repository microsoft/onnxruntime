// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/layer_norm.h"
#include "orttraining/training_ops/cuda/nn/layer_norm_impl.h"

#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_GRADIENT_KERNEL_TYPED(T, T1, U)                                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(LayerNormalizationGrad, kMSDomain, 1, T##_##T1##_##U, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                                 \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                    \
                                    .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                  \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()),                   \
                                LayerNormGrad<T, T1, U, false>);                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(InvertibleLayerNormalizationGrad, kMSDomain, 1, T##_##T1##_##U,               \
                                kCudaExecutionProvider,                                                       \
                                (*KernelDefBuilder::Create())                                                 \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                    \
                                    .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                  \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()),                   \
                                InvertibleLayerNormGrad<T, T1, U>);                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(SimplifiedLayerNormalizationGrad, kMSDomain, 1, T##_##T1##_##U,               \
                                kCudaExecutionProvider,                                                       \
                                (*KernelDefBuilder::Create())                                                 \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                    \
                                    .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                  \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()),                   \
                                LayerNormGrad<T, T1, U, true>);

REGISTER_GRADIENT_KERNEL_TYPED(float, float, float)
REGISTER_GRADIENT_KERNEL_TYPED(double, double, double)
REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16, MLFloat16, float)
REGISTER_GRADIENT_KERNEL_TYPED(float, MLFloat16, float)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
REGISTER_GRADIENT_KERNEL_TYPED(BFloat16, BFloat16, float)
#endif

template <typename T, typename T1, typename U, bool simplified>
LayerNormGrad<T, T1, U, simplified>::LayerNormGrad(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
}

template <typename T, typename T1, typename U, bool simplified>
Status LayerNormGrad<T, T1, U, simplified>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<U>::MappedType CudaU;
  // Inputs
  int input_index = 0;
  const Tensor* Y_grad = p_op_kernel_context->Input<Tensor>(input_index++);
  const Tensor* X = p_op_kernel_context->Input<Tensor>(input_index++);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(input_index++);
  const Tensor* mean;
  if (!simplified) {
    mean = p_op_kernel_context->Input<Tensor>(input_index++);
  }
  const Tensor* inv_std_var = p_op_kernel_context->Input<Tensor>(input_index);

  auto Y_grad_data = reinterpret_cast<const CudaT*>(Y_grad->template Data<T>());
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT1*>(scale->template Data<T1>());
  auto mean_data = simplified ? nullptr : reinterpret_cast<const CudaU*>(mean->template Data<U>());
  auto inv_std_var_data = reinterpret_cast<const CudaU*>(inv_std_var->template Data<U>());

  const TensorShape& x_shape = X->Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
  auto n1 = x_shape.SizeToDimension(axis);
  auto n2 = x_shape.SizeFromDimension(axis);
  ORT_ENFORCE(n2 != 1, "n2 should not be 1");

  // Outputs
  Tensor* X_grad = p_op_kernel_context->Output(0, x_shape);
  auto X_grad_data = reinterpret_cast<CudaT*>(X_grad->template MutableData<T>());
  Tensor* scale_grad = p_op_kernel_context->Output(1, scale->Shape());
  auto scale_grad_data = reinterpret_cast<CudaT1*>(scale_grad->template MutableData<T1>());
  CudaT1* bias_grad_data = nullptr;
  if (!simplified) {
    Tensor* bias_grad = p_op_kernel_context->Output(2, scale->Shape());
    bias_grad_data = reinterpret_cast<CudaT1*>(bias_grad->template MutableData<T1>());
  }

  const int part_size = 16;
  auto part_grad_gamma = GetScratchBuffer<CudaU>(part_size * n2);
  auto part_grad_beta = GetScratchBuffer<CudaU>(part_size * n2);

  HostLayerNormGradient<CudaT, CudaT1, CudaU, simplified>(
      GetDeviceProp(), Stream(), Y_grad_data, X_data, reinterpret_cast<const CudaT*>(NULL), scale_data,
      reinterpret_cast<const CudaT1*>(NULL), mean_data, inv_std_var_data, n1, n2, X_grad_data, scale_grad_data,
      bias_grad_data, part_grad_gamma.get(), part_grad_beta.get(), part_size);
  return Status::OK();
}

template <typename T, typename T1, typename U>
InvertibleLayerNormGrad<T, T1, U>::InvertibleLayerNormGrad(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
}

template <typename T, typename T1, typename U>
Status InvertibleLayerNormGrad<T, T1, U>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<U>::MappedType CudaU;
  // Inputs
  const Tensor* Y_grad = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* Y = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* bias = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* inv_std_var = p_op_kernel_context->Input<Tensor>(4);

  auto Y_grad_data = reinterpret_cast<const CudaT*>(Y_grad->template Data<T>());
  auto Y_data = reinterpret_cast<const CudaT*>(Y->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT1*>(scale->template Data<T1>());
  auto bias_data = reinterpret_cast<const CudaT1*>(bias->template Data<T1>());
  auto inv_std_var_data = reinterpret_cast<const CudaU*>(inv_std_var->template Data<U>());

  const TensorShape& y_shape = Y->Shape();
  const TensorShape& x_shape = y_shape;
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
  auto n1 = x_shape.SizeToDimension(axis);
  auto n2 = x_shape.SizeFromDimension(axis);
  ORT_ENFORCE(n2 != 1, "n2 should not be 1");

  // Outputs
  Tensor* X_grad = p_op_kernel_context->Output(0, x_shape);
  auto X_grad_data = reinterpret_cast<CudaT*>(X_grad->template MutableData<T>());

  Tensor* scale_grad = p_op_kernel_context->Output(1, scale->Shape());
  Tensor* bias_grad = p_op_kernel_context->Output(2, scale->Shape());
  auto scale_grad_data = reinterpret_cast<CudaT1*>(scale_grad->template MutableData<T1>());
  auto bias_grad_data = reinterpret_cast<CudaT1*>(bias_grad->template MutableData<T1>());

  const int part_size = 16;
  auto part_grad_gamma = GetScratchBuffer<CudaU>(part_size * n2);
  auto part_grad_beta = GetScratchBuffer<CudaU>(part_size * n2);

  HostLayerNormGradient<CudaT, CudaT1, CudaU, false>(
      GetDeviceProp(), Stream(), Y_grad_data, reinterpret_cast<const CudaT*>(NULL), Y_data, scale_data, bias_data,
      reinterpret_cast<const CudaU*>(NULL), inv_std_var_data, n1, n2, X_grad_data, scale_grad_data, bias_grad_data,
      part_grad_gamma.get(), part_grad_beta.get(), part_size);
  return Status::OK();
}

}  //namespace cuda
}  // namespace onnxruntime
