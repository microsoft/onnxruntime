// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/instance_norm.h"
#include "core/providers/cpu/nn/instance_norm_helper.h"
#include "core/util/math_cpuonly.h"
using namespace ::onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    InstanceNormalization,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    InstanceNorm<float>);

template <>
Status InstanceNorm<float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* input = p_op_kernel_context->Input<Tensor>(0);
  const auto* scale = p_op_kernel_context->Input<Tensor>(1);
  const auto* B = p_op_kernel_context->Input<Tensor>(2);

  ORT_RETURN_IF_ERROR(InstanceNormHelper::ValidateInputs(input, scale, B));
  const int64_t N = input->Shape().GetDims()[0];
  const int64_t C = input->Shape().GetDims()[1];
  const int64_t W = input->Shape().SizeFromDimension(2);

  const TensorShape& x_shape = input->Shape();
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);

  for (auto i = 0; i < N * C; ++i) {
    ConstEigenVectorArrayMap<float> Xi(input->Data<float>() + W * i, W);
    const float Xi_mean = Xi.mean();
    const float squared_norm = (Xi - Xi_mean).matrix().squaredNorm();
    const float inv_stdev = 1.0f / std::sqrt(squared_norm / W + epsilon_);
    EigenVectorArrayMap<float> Yi(Y->MutableData<float>() + W * i, W);
    const float channel_scale = inv_stdev * scale->Data<float>()[i % C];
    const float channel_shift = B->Data<float>()[i % C] - Xi_mean * channel_scale;
    Yi = Xi * channel_scale + channel_shift;
  }

  return Status::OK();
}
}  // namespace onnxruntime
