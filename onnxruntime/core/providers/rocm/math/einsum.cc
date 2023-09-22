// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum.h"

namespace onnxruntime {

// This function must exist due to the C++ base class constructor needing this to be defined for the vtable, but it is never called.
Status Einsum::DeviceCompute(OpKernelContext* /*context*/, const std::vector<const Tensor*>& /*inputs*/,
                             AllocatorPtr /*allocator*/, concurrency::ThreadPool* /*tp*/) const {
  assert(false);
  return Status::OK();
}

namespace rocm {

ONNX_OPERATOR_KERNEL_EX(
    Einsum,
    kOnnxDomain,
    12,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>()}),
    Einsum);

Status Einsum::Compute(OpKernelContext* context) const {
  return onnxruntime::Einsum::Compute(context);
}

Status Einsum::DeviceCompute(OpKernelContext* context, const std::vector<const Tensor*>& inputs,
                             AllocatorPtr allocator, concurrency::ThreadPool* tp) const {
  auto* stream = context->GetComputeStream();
  ORT_RETURN_IF(!stream, "stream is null");
  auto* rocm_stream = static_cast<RocmStream*>(stream);
  rocblas_handle rocblas_handle = rocm_stream ? rocm_stream->rocblas_handle_ : nullptr;
  EinsumOp::EinsumRocmAssets einsum_rocm_assets(rocblas_handle, rocm_ep_, stream, Info().GetAllocator(OrtMemType::OrtMemTypeDefault));

  // EinsumComputePreprocessor section -
  auto einsum_compute_preprocessor = EinsumComputePreprocessor::Create(*einsum_equation_preprocessor_, inputs, allocator,
                                                                       &einsum_rocm_assets);

  einsum_compute_preprocessor->SetDeviceHelpers(EinsumOp::DeviceHelpers::RocmDeviceHelpers::Diagonal,
                                                EinsumOp::DeviceHelpers::RocmDeviceHelpers::Transpose);
  // Compute all required metadata to be used at Einsum compute time and return error status code if one was generated
  ORT_RETURN_IF_ERROR(einsum_compute_preprocessor->Run());

  // EinsumComputeProcessor section -
  if (inputs[0]->IsDataType<float>()) {
    auto einsum_compute_processor = EinsumTypedComputeProcessor<float>::Create(context, allocator, tp,
                                                                               *einsum_compute_preprocessor,
                                                                               &einsum_rocm_assets);

    einsum_compute_processor->SetDeviceHelpers(EinsumOp::DeviceHelpers::RocmDeviceHelpers::Transpose,
                                               EinsumOp::DeviceHelpers::RocmDeviceHelpers::MatMul<float>,
                                               EinsumOp::DeviceHelpers::RocmDeviceHelpers::ReduceSum<float>,
                                               EinsumOp::DeviceHelpers::RocmDeviceHelpers::DataCopy);
    return einsum_compute_processor->Run();
  } else if (inputs[0]->IsDataType<MLFloat16>()) {
    auto einsum_compute_processor = EinsumTypedComputeProcessor<MLFloat16>::Create(context, allocator, tp,
                                                                                   *einsum_compute_preprocessor,
                                                                                   &einsum_rocm_assets);

    einsum_compute_processor->SetDeviceHelpers(EinsumOp::DeviceHelpers::RocmDeviceHelpers::Transpose,
                                               EinsumOp::DeviceHelpers::RocmDeviceHelpers::MatMul<MLFloat16>,
                                               EinsumOp::DeviceHelpers::RocmDeviceHelpers::ReduceSum<MLFloat16>,
                                               EinsumOp::DeviceHelpers::RocmDeviceHelpers::DataCopy);
    return einsum_compute_processor->Run();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Einsum op: An implementation for the input type ",
                         inputs[0]->DataType(), " is not supported yet");
}

}  // namespace rocm

}  // namespace onnxruntime
