// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "layer_norm.h"
#include "layer_norm_impl.h"

#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, U)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LayerNormalization,                                         \
      kOnnxDomain,                                                \
      1,                                                          \
      T##_##U,                                                    \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()), \
      LayerNorm<T, U>);

REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(double, double)
REGISTER_KERNEL_TYPED(MLFloat16, float)

template <typename T, typename U>
LayerNorm<T, U>::LayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
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
  CudaU* mean_data = nullptr;
  if (mean != nullptr) {
    mean_data = reinterpret_cast<CudaU*>(mean->template MutableData<U>());
  }

  CudaU* inv_var_data = nullptr;
  if (var != nullptr) {
    inv_var_data = reinterpret_cast<CudaU*>(var->template MutableData<U>());
  }

  HostApplyLayerNorm(GetDeviceProp(), Y_data, mean_data, inv_var_data, X_data, n1, n2, epsilon_, scale_data, bias_data);
  return Status::OK();
}

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
