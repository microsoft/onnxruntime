/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#include "core/providers/cpu/nn/batch_norm.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"

namespace onnxruntime {
// spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    BatchNormalization,
    7,
    9,
    KernelDefBuilder().TypeConstraint("X", DataTypeImpl::GetTensorType<float>()).TypeConstraint("scale", DataTypeImpl::GetTensorType<float>()).TypeConstraint("B", DataTypeImpl::GetTensorType<float>()).TypeConstraint("mean", DataTypeImpl::GetTensorType<float>()).TypeConstraint("var", DataTypeImpl::GetTensorType<float>()),
    BatchNorm<float>);

template <>
Status BatchNorm<float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* X = p_op_kernel_context->Input<Tensor>(0);
  const auto* scale = p_op_kernel_context->Input<Tensor>(1);
  const auto* B = p_op_kernel_context->Input<Tensor>(2);
  const auto* mean = p_op_kernel_context->Input<Tensor>(3);
  const auto* var = p_op_kernel_context->Input<Tensor>(4);

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var));

  const TensorShape& x_shape = X->Shape();
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);

  const auto& dims_vec = x_shape.GetDims();
  const size_t N = dims_vec[0];
  const size_t C = dims_vec[1];  // assume NCHW as per the spec

  // calculate sample_size
  size_t sample_size = 1;
  for (size_t i = 2; i < dims_vec.size(); ++i) {
    sample_size *= dims_vec[i];
  }

  ConstEigenVectorArrayMap<float> scale_arr(scale->template Data<float>(), C);
  ConstEigenVectorArrayMap<float> bias_arr(B->template Data<float>(), C);

  // Regardless of training or testing, we will apply the estimated mean
  // and standard deviation to the input. For testing, they are
  // specified directly by the input, and for training, they are computed
  // by the op.
  Eigen::Array<float, Eigen::Dynamic, 1> inv_std(C);
  ConstEigenVectorArrayMap<float> var_arr(var->template Data<float>(), C);
  inv_std = (var_arr + epsilon_).sqrt().inverse();
  ConstEigenVectorArrayMap<float> mean_arr(mean->template Data<float>(), C);
  // We can fuse the output computation as follows:
  //   ((x - est_mean) * (inv_var) * scale + bias
  // to
  //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  Eigen::Array<float, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
  Eigen::Array<float, Eigen::Dynamic, 1> new_bias = bias_arr - mean_arr * new_scale;
  EigenArrayMap<float> Y_arr(Y->template MutableData<float>(), sample_size, N * C);
  ConstEigenArrayMap<float> X_arr(X->template Data<float>(), sample_size, N * C);
  for (size_t nc = 0; nc < N * C; ++nc) {
    Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
  }

  return Status::OK();
}
}  // namespace onnxruntime
