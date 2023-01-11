// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/nn/layer_norm.h"
#include "core/providers/cuda/nn/layer_norm_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, U)                                                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(LayerNormalization, kOnnxDomain, 17, T, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                   \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()),     \
                                LayerNorm<T, U, T, false>);

REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(double, float)
REGISTER_KERNEL_TYPED(MLFloat16, float)
REGISTER_KERNEL_TYPED(BFloat16, float)

template <typename T, typename U, typename V, bool simplified>
LayerNorm<T, U, V, simplified>::LayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
}

template <typename T, typename U, typename V, bool simplified>
Status LayerNorm<T, U, V, simplified>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;
  typedef typename ToCudaType<V>::MappedType CudaV;
  // Inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(1);
  const Tensor* bias = ctx->Input<Tensor>(2);

  auto X_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto scale_data = reinterpret_cast<const CudaV*>(scale->Data<V>());
  auto bias_data = (simplified || (nullptr == bias)) ? nullptr : reinterpret_cast<const CudaV*>(bias->Data<V>());

  const TensorShape& x_shape = X->Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());

  int n1 = gsl::narrow<int>(x_shape.SizeToDimension(axis));
  int n2 = gsl::narrow<int>(x_shape.SizeFromDimension(axis));

  const auto scale_size = scale->Shape().Size();
  const auto bias_size = (bias_data) ? bias->Shape().Size() : 0;
  if (n2 == 1 || scale_size != n2 || (bias_data && bias_size != n2)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Size of X.shape()[axis:] == ", n2,
                           ". Size of scale and bias (if provided) must match this "
                           "and the size must not be 1. Got scale size of ",
                           scale_size, " and bias size of ", bias_size);
  }

  // Outputs
  Tensor* Y = ctx->Output(0, x_shape);
  auto Y_data = reinterpret_cast<CudaV*>(Y->MutableData<V>());

  // Mean and variance
  std::vector<int64_t> mean_inv_std_var_dim;
  for (int i = 0; i < static_cast<int>(x_shape.NumDimensions()); ++i) {
    if (i < axis) {
      mean_inv_std_var_dim.emplace_back(x_shape.GetDims()[i]);
    } else {
      mean_inv_std_var_dim.emplace_back(1);
    }
  }
  int output_index = 1;

  CudaU* mean_data = nullptr;

  if (!simplified) {
    Tensor* mean = ctx->Output(output_index++, TensorShape(mean_inv_std_var_dim));
    if (mean != nullptr) {
      mean_data = reinterpret_cast<CudaU*>(mean->MutableData<U>());
    }
  }

  CudaU* inv_var_data = nullptr;
  Tensor* var = ctx->Output(output_index, TensorShape(mean_inv_std_var_dim));
  if (var != nullptr) {
    inv_var_data = reinterpret_cast<CudaU*>(var->MutableData<U>());
  }

  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  HostApplyLayerNorm<CudaT, CudaU, CudaV, simplified>(GetDeviceProp(), Stream(ctx), Y_data, mean_data, inv_var_data,
                                                      X_data, n1, n2, epsilon_, scale_data, bias_data);
  return Status::OK();
}

#if !defined(DISABLE_CONTRIB_OPS)
#define LAYERNORM_IMPL(T, U, V, simplified) \
  template class LayerNorm<T, U, V, simplified>;

// contrib op usage
LAYERNORM_IMPL(float, float, float, false)
LAYERNORM_IMPL(double, double, double, false)
LAYERNORM_IMPL(MLFloat16, float, MLFloat16, false)
LAYERNORM_IMPL(float, float, MLFloat16, false)
LAYERNORM_IMPL(MLFloat16, float, float, false)
LAYERNORM_IMPL(BFloat16, float, BFloat16, false)

LAYERNORM_IMPL(float, float, float, true)
LAYERNORM_IMPL(double, double, double, true)
LAYERNORM_IMPL(MLFloat16, float, MLFloat16, true)
LAYERNORM_IMPL(float, float, MLFloat16, true)
LAYERNORM_IMPL(MLFloat16, float, float, true)
LAYERNORM_IMPL(BFloat16, float, BFloat16, true)
#endif
}  // namespace cuda
}  // namespace onnxruntime
