// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/rms_norm.h"
#include "core/providers/cuda/nn/layer_norm.h"
#include "core/providers/cuda/nn/layer_norm_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/nn/layer_norm_helper.h"

namespace onnxruntime {
namespace cuda {

// RMSNorm uses LayerNorm kernel, which only supports X and scale both
// being the same data type.
#define REGISTER_KERNEL_TYPED(T)                                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(RMSNormalization, kOnnxDomain, 23, T, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                 \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())    \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<T>()),   \
                                RMSNorm<T, float, T, true>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T, typename U, typename V, bool simplified>
RMSNorm<T, U, V, simplified>::RMSNorm(const OpKernelInfo& op_kernel_info)
    : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
}

template <typename T, typename U, typename V, bool simplified>
Status RMSNorm<T, U, V, simplified>::ComputeInternal(OpKernelContext* ctx) const {
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
  auto x_num_dims = x_shape.NumDimensions();
  const int64_t axis = HandleNegativeAxis(axis_, x_num_dims);

  const TensorShape& scale_shape = scale->Shape();
  const TensorShape& bias_shape = bias_data ? bias->Shape() : TensorShape();

  LayerNormParams params;
  ORT_RETURN_IF_ERROR(LayerNormHelper::CheckInputs(x_shape, scale_shape, bias_shape, bias_data != nullptr, axis, params));

  // Outputs
  Tensor* Y = ctx->Output(0, x_shape);
  auto Y_data = reinterpret_cast<CudaV*>(Y->MutableData<V>());

  // Mean and variance
  std::vector<int64_t> mean_inv_std_var_dim;
  for (int i = 0; i < static_cast<int>(x_num_dims); ++i) {
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

  HostApplyLayerNorm<CudaT, CudaU, CudaV, simplified>(
      GetDeviceProp(), Stream(ctx), Y_data, mean_data, inv_var_data, X_data,
      onnxruntime::narrow<int>(params.num_rows), onnxruntime::narrow<int>(params.norm_size), epsilon_,
      scale_data, bias_data,
      onnxruntime::narrow<int>(params.broadcast_param));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
