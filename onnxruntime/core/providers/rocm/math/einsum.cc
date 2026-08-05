// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum.h"
#include "core/providers/cpu/math/einsum_utils/einsum_compute_preprocessor.h"
#include "core/providers/cpu/math/einsum_utils/einsum_typed_compute_processor.h"

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
  hipblasHandle_t hipblas_handle = rocm_stream ? rocm_stream->hipblas_handle_ : nullptr;
  EinsumOp::EinsumRocmAssets einsum_rocm_assets(hipblas_handle, rocm_ep_, stream, Info().GetAllocator(OrtMemType::OrtMemTypeDefault));

  EinsumComputePreprocessor einsum_compute_preprocessor(*einsum_equation_preprocessor_, inputs, allocator,
                                                        &einsum_rocm_assets);
  einsum_compute_preprocessor.SetDeviceHelpers(EinsumOp::DeviceHelpers::RocmDeviceHelpers::Diagonal,
                                               EinsumOp::DeviceHelpers::RocmDeviceHelpers::Transpose,
                                               EinsumOp::DeviceHelpers::RocmDeviceHelpers::CreateTensor);
  ORT_RETURN_IF_ERROR(einsum_compute_preprocessor.Run());

  if (inputs[0]->IsDataType<float>()) {
    EinsumTypedComputeProcessor<float> einsum_compute_processor(context, allocator, tp, nullptr,
                                                                einsum_compute_preprocessor,
                                                                &einsum_rocm_assets);
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::RocmDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::RocmDeviceHelpers::MatMul<float>,
                                              EinsumOp::DeviceHelpers::RocmDeviceHelpers::ReduceSum<float>,
                                              EinsumOp::DeviceHelpers::RocmDeviceHelpers::DataCopy,
                                              EinsumOp::DeviceHelpers::RocmDeviceHelpers::ZeroBuffer,
                                              EinsumOp::DeviceHelpers::RocmDeviceHelpers::CreateTensor);
    return einsum_compute_processor.Run();
  } else if (inputs[0]->IsDataType<MLFloat16>()) {
    EinsumTypedComputeProcessor<MLFloat16> einsum_compute_processor(context, allocator, tp, nullptr,
                                                                    einsum_compute_preprocessor,
                                                                    &einsum_rocm_assets);
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::RocmDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::RocmDeviceHelpers::MatMul<MLFloat16>,
                                              EinsumOp::DeviceHelpers::RocmDeviceHelpers::ReduceSum<MLFloat16>,
                                              EinsumOp::DeviceHelpers::RocmDeviceHelpers::DataCopy,
                                              EinsumOp::DeviceHelpers::RocmDeviceHelpers::ZeroBuffer,
                                              EinsumOp::DeviceHelpers::RocmDeviceHelpers::CreateTensor);
    return einsum_compute_processor.Run();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Einsum op: An implementation for the input type ",
                         inputs[0]->DataType(), " is not supported yet");
}

}  // namespace rocm

}  // namespace onnxruntime
