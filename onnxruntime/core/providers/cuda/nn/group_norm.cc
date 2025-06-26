// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/nn/group_norm.h"
#include "core/providers/cuda/nn/group_norm_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

// Opset 18-20 registrations (without stash_type)
#define REGISTER_CUDA_KERNEL_TYPED_VERSIONED(T)                                                     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(GroupNormalization, kOnnxDomain, 18, 20, T, kCudaExecutionProvider, \
                                          (*KernelDefBuilder::Create())                             \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                          GroupNorm<T, float>);

// Opset 21+ registrations (with stash_type)
#define REGISTER_CUDA_KERNEL_TYPED_21(T)                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(GroupNormalization, kOnnxDomain, 21, T, kCudaExecutionProvider,    \
                                (*KernelDefBuilder::Create())                                       \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),         \
                                GroupNorm<T, float>);

REGISTER_CUDA_KERNEL_TYPED_VERSIONED(float)
REGISTER_CUDA_KERNEL_TYPED_VERSIONED(double)
REGISTER_CUDA_KERNEL_TYPED_VERSIONED(MLFloat16)
REGISTER_CUDA_KERNEL_TYPED_VERSIONED(BFloat16)

REGISTER_CUDA_KERNEL_TYPED_21(float)
REGISTER_CUDA_KERNEL_TYPED_21(double)
REGISTER_CUDA_KERNEL_TYPED_21(MLFloat16)
REGISTER_CUDA_KERNEL_TYPED_21(BFloat16)

template <typename T, typename U>
GroupNorm<T, U>::GroupNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
  
  ORT_ENFORCE(op_kernel_info.GetAttr("num_groups", &num_groups_).IsOK());
  
  // stash_type is optional in opset 21, default to 1 (float32)
  if (!op_kernel_info.GetAttr("stash_type", &stash_type_).IsOK()) {
    stash_type_ = 1;
  }
}

template <typename T, typename U>
Status GroupNorm<T, U>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  
  // Inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(1);
  const Tensor* bias = ctx->Input<Tensor>(2);

  auto X_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->Data<T>());
  auto bias_data = reinterpret_cast<const CudaT*>(bias->Data<T>());

  const auto& x_shape = X->Shape();
  const int64_t N = x_shape[0];  // batch size
  const int64_t C = x_shape[1];  // channels
  
  // Validate that channels are divisible by num_groups
  ORT_RETURN_IF_NOT(C % num_groups_ == 0, "Number of channels must be divisible by num_groups");
  
  // Calculate spatial dimensions (H*W*... for everything after batch and channel dims)
  int64_t spatial_size = 1;
  for (size_t i = 2; i < x_shape.NumDimensions(); ++i) {
    spatial_size *= x_shape[i];
  }
  
  Tensor* Y = ctx->Output(0, x_shape);
  auto Y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());

  return GroupNormImpl<CudaT, U>(
      Stream(ctx),
      X_data,
      scale_data,
      bias_data,
      Y_data,
      N,
      C,
      spatial_size,
      num_groups_,
      stash_type_,
      epsilon_);
}

}  // namespace cuda
}  // namespace onnxruntime