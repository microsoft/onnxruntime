// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/rms_norm.h"
#include "core/providers/cuda/nn/layer_norm.h"
#include "core/providers/cuda/nn/layer_norm_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/nn/layer_norm_helper.h"
#include <vector>

namespace onnxruntime {
namespace cuda {

// For T and V both are double, we need U to be double
template <typename T, typename V>
using MeanVarType = typename std::conditional<
    std::is_same<T, double>::value && std::is_same<V, double>::value,
    double,
    float>::type;

// RMSNorm uses LayerNorm kernel, which only supports X and scale both
// being the same data type.
#define REGISTER_KERNEL_TYPED(T, V)                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(RMSNormalization, kOnnxDomain, 23, T##_##V, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                       \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())          \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<V>()),         \
                                RMSNorm<T, MeanVarType<T, V>, V>);

REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(float, MLFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, float)
REGISTER_KERNEL_TYPED(BFloat16, BFloat16)
REGISTER_KERNEL_TYPED(double, double)

// The following code is shared from "core/providers/cuda/nn/layer_norm.cc".
// It is used to implement the RMSNorm kernel, which is a simplified version of LayerNorm.
template <typename T, typename U, typename V>
RMSNorm<T, U, V>::RMSNorm(const OpKernelInfo& op_kernel_info)
    : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
}

template <typename T, typename U, typename V>
Status RMSNorm<T, U, V>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;
  typedef typename ToCudaType<V>::MappedType CudaV;
  // Inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(1);

  auto X_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto scale_data = reinterpret_cast<const CudaV*>(scale->Data<V>());
  auto bias_data = nullptr;

  const TensorShape& x_shape = X->Shape();
  auto x_num_dims = x_shape.NumDimensions();
  const int64_t axis = HandleNegativeAxis(axis_, x_num_dims);

  const TensorShape& scale_shape = scale->Shape();
  const TensorShape& bias_shape = TensorShape();

  LayerNormParams params;
  ORT_RETURN_IF_ERROR(LayerNormHelper::CheckInputs(x_shape, scale_shape, bias_shape, bias_data != nullptr, axis, params));

  // Outputs
  Tensor* Y = ctx->Output(0, x_shape);
  auto Y_data = reinterpret_cast<CudaV*>(Y->MutableData<V>());

  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  // For RMSNorm, we don't need mean and inv_var data, so we can pass nullptr.
  CudaU* mean_data = nullptr;
  CudaU* inv_var_data = nullptr;

  // simplified should always be true for RMSNorm
  HostApplyLayerNorm<CudaT, CudaU, CudaV, true>(
      GetDeviceProp(), Stream(ctx), Y_data, mean_data, inv_var_data, X_data,
      onnxruntime::narrow<int>(params.num_rows), onnxruntime::narrow<int>(params.norm_size), epsilon_,
      scale_data, bias_data,
      onnxruntime::narrow<int>(params.broadcast_param));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
