// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/unsqueeze.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    1, 10,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Unsqueeze);

// explicitly support negative axis
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    11, 12,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Unsqueeze);

// support bfloat16
ONNX_OPERATOR_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Unsqueeze);

Status Unsqueeze::ComputeInternal(OpKernelContext* ctx) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, p));

  const void* input = p.input_tensor->DataRaw();
  void* output = p.output_tensor->MutableDataRaw();
  if (input == output)
    return Status::OK();

  auto count = p.input_tensor->Shape().Size();
  auto element_bytes = p.input_tensor->DataType()->Size();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output, input, count * element_bytes, cudaMemcpyDeviceToDevice));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
