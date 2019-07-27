// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "layer_norm.h"

namespace onnxruntime {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LayerNormalization,                                         \
      kOnnxDomain,                                                \
      9,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LayerNorm<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

template <typename T>
LayerNorm<T>::LayerNorm(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
}

template <typename T>
Status LayerNorm<T>::Compute(OpKernelContext* p_op_kernel_context) const {
  // Inputs
  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* bias = p_op_kernel_context->Input<Tensor>(2);
  auto X_data = X->template Data<T>();
  auto scale_data = scale->template Data<T>();
  auto bias_data = bias->template Data<T>();

  const TensorShape& x_shape = X->Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
  auto N = x_shape.SizeToDimension(axis);
  auto M = x_shape.SizeFromDimension(axis);

  // Compute Mean and Variance
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
  auto mean_data = mean->template MutableData<T>();
  auto inv_std_var_data = inv_std_var->template MutableData<T>();
  std::memset(mean_data, 0, sizeof(T) * N);
  std::memset(inv_std_var_data, 0, sizeof(T) * N);

  ConstEigenArrayMap<T> X_arr(X_data, M, N);
  for (int i = 0; i < N; ++i) {
    mean_data[i] = X_arr.col(i).mean();
    inv_std_var_data[i] = X_arr.col(i).square().mean() - mean_data[i] * mean_data[i];
  }

  // Compute Y = ((x - mean) * (inv_var) * scale + bias
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);
  auto Y_data = Y->template MutableData<T>();
  EigenArrayMap<T> Y_arr(Y_data, M, N);

  ConstEigenVectorArrayMap<T> mean_arr(mean_data, N);
  EigenVectorArrayMap<T> inv_std_var_arr(inv_std_var_data, N);
  inv_std_var_arr = (inv_std_var_arr + epsilon_).sqrt().inverse();

  Y_arr = (X_arr.rowwise() - mean_arr.transpose()).rowwise() * inv_std_var_arr.transpose(); 

  ConstEigenVectorArrayMap<T> scale_arr(scale_data, M);
  ConstEigenVectorArrayMap<T> bias_arr(bias_data, M);
  Y_arr = (Y_arr.colwise() * scale_arr).colwise() + bias_arr;

  return Status::OK();
}

}  // namespace onnxruntime
