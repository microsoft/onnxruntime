// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/layer_norm.h"
#include "orttraining/training_ops/cuda/nn/layer_norm_impl.h"

#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_KERNEL_TYPED(T, U)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LayerNormalization,                                         \
      kMSDomain,                                                  \
      1,                                                          \
      T##_##U,                                                    \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()), \
      LayerNorm<T, U>);

REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(double, float)
REGISTER_KERNEL_TYPED(MLFloat16, float)

template <typename T, typename U>
LayerNorm<T, U>::LayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = ClampCudnnBatchNormEpsilon(tmp_epsilon);
}

template <typename T, typename U>
Status LayerNorm<T, U>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;
  //Inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(1);
  const Tensor* bias = ctx->Input<Tensor>(2);

  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->template Data<T>());
  auto bias_data = reinterpret_cast<const CudaT*>(bias->template Data<T>());

  const TensorShape& x_shape = X->Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());

  auto n1 = x_shape.SizeToDimension(axis);
  auto n2 = x_shape.SizeFromDimension(axis);

  ORT_ENFORCE(n2 != 1, "n2 should not be 1");

  // Outputs
  Tensor* Y = ctx->Output(0, x_shape);
  auto Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  //Mean and variance
  std::vector<int64_t> mean_inv_std_var_dim;
  for (int i = 0; i < static_cast<int>(x_shape.NumDimensions()); ++i) {
    if (i < axis) {
      mean_inv_std_var_dim.emplace_back(x_shape.GetDims()[i]);
    } else {
      mean_inv_std_var_dim.emplace_back(1);
    }
  }
  Tensor* mean = ctx->Output(1, TensorShape(mean_inv_std_var_dim));
  Tensor* var = ctx->Output(2, TensorShape(mean_inv_std_var_dim));
  auto mean_data = reinterpret_cast<CudaU*>(mean->template MutableData<U>());
  auto inv_var_data = reinterpret_cast<CudaU*>(var->template MutableData<U>());

  HostApplyLayerNorm(Y_data, mean_data, inv_var_data, X_data, n1, n2, epsilon_, scale_data, bias_data);
  return Status::OK();
}

#define REGISTER_GRADIENT_KERNEL_TYPED(T, U)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LayerNormalizationGrad,                                     \
      kOnnxDomain,                                                \
      9,                                                          \
      T##_##U,                                                    \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()), \
      LayerNormGrad<T, U>);
REGISTER_GRADIENT_KERNEL_TYPED(float, float)
REGISTER_GRADIENT_KERNEL_TYPED(double, float)
REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16, float)

template <typename T, typename U>
LayerNormGrad<T, U>::LayerNormGrad(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
}

template <typename T, typename U>
Status LayerNormGrad<T, U>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;
  // Inputs
  const Tensor* Y_grad = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* X = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* mean = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* inv_std_var = p_op_kernel_context->Input<Tensor>(4);

  auto Y_grad_data = reinterpret_cast<const CudaT*>(Y_grad->template Data<T>());
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->template Data<T>());
  auto mean_data = reinterpret_cast<const CudaU*>(mean->template Data<U>());
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
  Tensor* bias_grad = p_op_kernel_context->Output(2, scale->Shape());
  auto scale_grad_data = reinterpret_cast<CudaT*>(scale_grad->template MutableData<T>());
  auto bias_grad_data = reinterpret_cast<CudaT*>(bias_grad->template MutableData<T>());

  const int part_size = 16;
  auto part_grad_gamma = GetScratchBuffer<CudaU>(part_size * n2);
  auto part_grad_beta = GetScratchBuffer<CudaU>(part_size * n2);

  HostLayerNormGradient(Y_grad_data, mean_data, inv_std_var_data, X_data, n1, n2, scale_data, X_grad_data, scale_grad_data, bias_grad_data,
                        part_grad_gamma.get(), part_grad_beta.get(), part_size);
  return Status::OK();
}

}  //namespace cuda
}  // namespace onnxruntime
