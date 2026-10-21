// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/matmul_sigmoid_gated_residual.h"

#include <limits>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

ONNX_OPERATOR_KERNEL_EX(
    MatMulSigmoidGatedResidual, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    MatMulSigmoidGatedResidual);

Status MatMulSigmoidGatedResidual::ComputeInternal(OpKernelContext* context) const {
  const Tensor* x = context->Input<Tensor>(0);
  const Tensor* weight = context->Input<Tensor>(1);
  const Tensor* shared = context->Input<Tensor>(2);
  const Tensor* routed = context->Input<Tensor>(3);
  const Tensor* residual = context->Input<Tensor>(4);
  const auto& shape = x->Shape();
  const auto& output_shape = shared->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 2 && output_shape.NumDimensions() == 2,
                    "X and shared must have rank 2");
  ORT_RETURN_IF_NOT(weight->Shape().NumDimensions() == 2 && weight->Shape()[0] == 1 &&
                        weight->Shape()[1] == shape[1],
                    "weight must have shape [1, K]");
  ORT_RETURN_IF_NOT(output_shape[0] == shape[0] && routed->Shape() == output_shape &&
                        residual->Shape() == output_shape,
                    "shared, routed and residual must have the same [M, N] shape and match X rows");
  ORT_RETURN_IF_NOT(shape[0] <= std::numeric_limits<int>::max() &&
                        shape[1] > 0 && shape[1] <= std::numeric_limits<int>::max() &&
                        output_shape[1] > 0 && output_shape[1] <= std::numeric_limits<int>::max(),
                    "M must fit int32; K and N must be positive and fit int32");
  Tensor* output = context->Output(0, output_shape);
  return LaunchMatMulSigmoidGatedResidual(
      Stream(context), x->DataRaw(), weight->DataRaw(), shared->DataRaw(), routed->DataRaw(),
      residual->DataRaw(), output->MutableDataRaw(), static_cast<int>(shape[0]),
      static_cast<int>(shape[1]), static_cast<int>(output_shape[1]));
}

}  // namespace onnxruntime::contrib::cuda
