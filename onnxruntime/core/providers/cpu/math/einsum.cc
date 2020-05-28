// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum.h"
#include "einsum_utils.h"

namespace onnxruntime {

// Credit: Implementation influenced by Torch's implementation at the time of writing

ONNX_CPU_OPERATOR_KERNEL(
    Einsum,
    12,
    KernelDefBuilder().TypeConstraint("T", std::vector<MLDataType>{
                                               DataTypeImpl::GetTensorType<float>(),
                                               DataTypeImpl::GetTensorType<double>(),
                                               DataTypeImpl::GetTensorType<int64_t>(),
                                               DataTypeImpl::GetTensorType<int32_t>()}),
    Einsum);

Status Einsum::Compute(OpKernelContext* context) const {
  int num_inputs = context->InputCount();
  if (num_inputs == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Einsum op: There must be atleast one input");
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
    return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION,
                           "There was a problem acquiring temporary memory allocator in Einsum op");
  }

  return DeviceCompute(context, inputs, allocator);
}

Status Einsum::DeviceCompute(OpKernelContext* context, const std::vector<const Tensor*>& inputs, AllocatorPtr allocator) const {
  // EinsumComputePreprocessor section -
  auto einsum_compute_preprocessor = EinsumComputePreprocessor(*einsum_equation_preprocessor_, inputs, allocator, nullptr);
  // Set device specific methods (CPU methods) to be used during pre-processing
  einsum_compute_preprocessor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CpuDeviceHelpers::Diagonal,
                                               EinsumOp::DeviceHelpers::CpuDeviceHelpers::Transpose);
  // Compute all required metadata to be used at Einsum compute time and return error status code if one was generated
  ORT_RETURN_IF_ERROR(einsum_compute_preprocessor.Run());

  // EinsumComputeProcessor section -
  if (inputs[0]->IsDataType<float>()) {
    auto einsum_compute_processor = EinsumTypedComputeProcessor<float>(context, allocator, einsum_compute_preprocessor, nullptr, nullptr);

    // Set device specific methods (CPU methods) to be used during processing
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CpuDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::MatMul<float>,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::ReduceSum<float>,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::DataCopy);
    return einsum_compute_processor.Run();
  } else if (inputs[0]->IsDataType<int32_t>()) {
    auto einsum_compute_processor = EinsumTypedComputeProcessor<int32_t>(context, allocator, einsum_compute_preprocessor, nullptr, nullptr);

    // Set device specific methods (CPU methods) to be used during processing
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CpuDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::MatMul<int32_t>,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::ReduceSum<int32_t>,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::DataCopy);

    return einsum_compute_processor.Run();
  } else if (inputs[0]->IsDataType<double>()) {
    auto einsum_compute_processor = EinsumTypedComputeProcessor<double>(context, allocator, einsum_compute_preprocessor, nullptr, nullptr);

    // Set device specific methods (CPU methods) to be used during processing
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CpuDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::MatMul<double>,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::ReduceSum<double>,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::DataCopy);
    return einsum_compute_processor.Run();
  } else if (inputs[0]->IsDataType<int64_t>()) {
    auto einsum_compute_processor = EinsumTypedComputeProcessor<int64_t>(context, allocator, einsum_compute_preprocessor, nullptr, nullptr);

    // Set device specific methods (CPU methods) to be used during processing
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CpuDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::MatMul<int64_t>,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::ReduceSum<int64_t>,
                                              EinsumOp::DeviceHelpers::CpuDeviceHelpers::DataCopy);

    return einsum_compute_processor.Run();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Einsum op: An implementation for the input type ",
                         inputs[0]->DataType(), " is not supported yet");
}

}  // namespace onnxruntime
