// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum.h"
#include "einsum_utils.h"

namespace onnxruntime {

// Credit: Implementation influenced by Torch's implementation at the time of writing

ONNX_CPU_OPERATOR_KERNEL(
    Einsum,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllNumericTensorTypes()),
    Einsum);

Status Einsum::Compute(OpKernelContext* context) const {
  int num_inputs = context->InputCount();
  if (num_inputs == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Einsum op: There must be atleast one input");
  }

  std::vector<const Tensor*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(context->Input<Tensor>(i));
  }

  // Get temp space allocator - we will use this to allocate memory for intermediate tensors
  AllocatorPtr allocator;
  auto status = context->GetTempSpaceAllocator(&allocator);
  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "There was a problem acquiring temporary space allocator in Einsum op");
  }

  // Instantiate EinsumComputePreprocessor
  auto einsum_compute_preprocessor = EinsumComputePreprocessor(*einsum_equation_preprocessor_, inputs, allocator);

  if (inputs[0]->IsDataType<float>()) {
    return EinsumTypedComputeProcessor<float>(context, allocator, einsum_compute_preprocessor);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Einsum op: An implementation for the input type ", inputs[0]->DataType(), " is not supported yet");
}

}  // namespace onnxruntime
