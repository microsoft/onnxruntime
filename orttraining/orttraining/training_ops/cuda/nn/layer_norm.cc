// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/layer_norm.h"
#include "orttraining/training_ops/cuda/nn/layer_norm_impl.h"

#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_GRADIENT_KERNEL_TYPED(T, U, V)                                                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(LayerNormalizationGrad, kMSDomain, 1, T##_##U##_##V, kCudaExecutionProvider,           \
                                (*KernelDefBuilder::Create())                                                          \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                             \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<U>())                             \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<V>()),                            \
                                LayerNormGrad<T, U, V, false>);                                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(InvertibleLayerNormalizationGrad, kMSDomain, 1, T##_##U##_##V, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                                          \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                             \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<U>())                             \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<V>()),                            \
                                InvertibleLayerNormGrad<T, U, V>);                                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(SimplifiedLayerNormalizationGrad, kMSDomain, 1, T##_##U##_##V, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                                          \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                             \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<U>())                             \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<V>()),                            \
                                LayerNormGrad<T, U, V, true>);

REGISTER_GRADIENT_KERNEL_TYPED(float, float, float)
REGISTER_GRADIENT_KERNEL_TYPED(double, double, double)
REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16, float, MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(float, float, MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16, float, float)
REGISTER_GRADIENT_KERNEL_TYPED(BFloat16, float, BFloat16)

template <typename T, typename U, typename V, bool simplified>
LayerNormGrad<T, U, V, simplified>::LayerNormGrad(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
}

template <typename T, typename U, typename V, bool simplified>
Status LayerNormGrad<T, U, V, simplified>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;
  typedef typename ToCudaType<V>::MappedType CudaV;
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

  auto Y_grad_data = reinterpret_cast<const CudaV*>(Y_grad->template Data<V>());
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaV*>(scale->template Data<V>());
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
  auto scale_grad_data = reinterpret_cast<CudaV*>(scale_grad->template MutableData<V>());
  CudaV* bias_grad_data = nullptr;
  if (!simplified) {
    Tensor* bias_grad = p_op_kernel_context->Output(2, scale->Shape());
    bias_grad_data = reinterpret_cast<CudaV*>(bias_grad->template MutableData<V>());
  }

  #ifndef USE_ROCM
  constexpr int part_size = 16;
  #else
  // Optimization for ROCm MI100
  constexpr int part_size = 64;
  #endif
  auto part_grad_gamma = GetScratchBuffer<CudaU>(part_size * n2, p_op_kernel_context->GetComputeStream());
  auto part_grad_beta = GetScratchBuffer<CudaU>(part_size * n2, p_op_kernel_context->GetComputeStream());

  HostLayerNormGradient<CudaT, CudaU, CudaV, simplified>(
      GetDeviceProp(), Stream(p_op_kernel_context), Y_grad_data, X_data, reinterpret_cast<const CudaV*>(NULL), scale_data,
      reinterpret_cast<const CudaV*>(NULL), mean_data, inv_std_var_data, n1, n2, X_grad_data, scale_grad_data,
      bias_grad_data, part_grad_gamma.get(), part_grad_beta.get(), part_size);
  return Status::OK();
}

template <typename T, typename U, typename V>
InvertibleLayerNormGrad<T, U, V>::InvertibleLayerNormGrad(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
}

template <typename T, typename U, typename V>
Status InvertibleLayerNormGrad<T, U, V>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;
  typedef typename ToCudaType<V>::MappedType CudaV;
  // Inputs
  const Tensor* Y_grad = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* Y = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* bias = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* inv_std_var = p_op_kernel_context->Input<Tensor>(4);

  auto Y_grad_data = reinterpret_cast<const CudaV*>(Y_grad->template Data<V>());
  auto Y_data = reinterpret_cast<const CudaV*>(Y->template Data<V>());
  auto scale_data = reinterpret_cast<const CudaV*>(scale->template Data<V>());
  auto bias_data = reinterpret_cast<const CudaV*>(bias->template Data<V>());
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
  auto scale_grad_data = reinterpret_cast<CudaV*>(scale_grad->template MutableData<V>());
  auto bias_grad_data = reinterpret_cast<CudaV*>(bias_grad->template MutableData<V>());

  #ifndef USE_ROCM
  constexpr int part_size = 16;
  #else
  // Optimization for ROCm MI100
  constexpr int part_size = 64;
  #endif
  auto part_grad_gamma = GetScratchBuffer<CudaU>(part_size * n2, p_op_kernel_context->GetComputeStream());
  auto part_grad_beta = GetScratchBuffer<CudaU>(part_size * n2, p_op_kernel_context->GetComputeStream());

  HostLayerNormGradient<CudaT, CudaU, CudaV, false>(
      GetDeviceProp(), Stream(p_op_kernel_context), Y_grad_data, reinterpret_cast<const CudaT*>(NULL), Y_data, scale_data, bias_data,
      reinterpret_cast<const CudaU*>(NULL), inv_std_var_data, n1, n2, X_grad_data, scale_grad_data, bias_grad_data,
      part_grad_gamma.get(), part_grad_beta.get(), part_size);
  return Status::OK();
}

}  //namespace cuda
}  // namespace onnxruntime
