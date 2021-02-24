// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum.h"

namespace onnxruntime {

namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Einsum,
    kOnnxDomain,
    12,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T",
                                      std::vector<MLDataType>{
                                          DataTypeImpl::GetTensorType<float>(),
                                          DataTypeImpl::GetTensorType<double>(),
                                          DataTypeImpl::GetTensorType<MLFloat16>()}),
    Einsum);

Status Einsum::Compute(OpKernelContext* context) const {
  return onnxruntime::Einsum::Compute(context);
}

Status Einsum::DeviceCompute(OpKernelContext* context, const std::vector<const Tensor*>& inputs,
                             AllocatorPtr allocator, concurrency::ThreadPool* tp) const {
  cublasHandle_t cublas_handle = cuda_ep_->PerThreadCublasHandle();

  EinsumOp::EinsumCudaAssets einsum_cuda_assets(cublas_handle, cuda_ep_);

  // EinsumComputePreprocessor section -
  auto einsum_compute_preprocessor = EinsumComputePreprocessor(*einsum_equation_preprocessor_, inputs, allocator,
                                                               &einsum_cuda_assets);

  einsum_compute_preprocessor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Diagonal,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose);
  // Compute all required metadata to be used at Einsum compute time and return error status code if one was generated
  ORT_RETURN_IF_ERROR(einsum_compute_preprocessor.Run());

  // EinsumComputeProcessor section -
  if (inputs[0]->IsDataType<float>()) {
    auto einsum_compute_processor = EinsumTypedComputeProcessor<float>(context, allocator, tp,
                                                                       einsum_compute_preprocessor,
                                                                       &einsum_cuda_assets);

    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<float>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<float>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy);
    return einsum_compute_processor.Run();
  } else if (inputs[0]->IsDataType<double>()) {
    auto einsum_compute_processor = EinsumTypedComputeProcessor<double>(context, allocator, tp,
                                                                        einsum_compute_preprocessor,
                                                                        &einsum_cuda_assets);

    // Set device specific methods (CPU methods) to be used during processing
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<double>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<double>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy);
    return einsum_compute_processor.Run();
  } else if (inputs[0]->IsDataType<MLFloat16>()) {
    auto einsum_compute_processor = EinsumTypedComputeProcessor<MLFloat16>(context, allocator, tp,
                                                                           einsum_compute_preprocessor,
                                                                           &einsum_cuda_assets);

    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<MLFloat16>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<MLFloat16>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy);
    return einsum_compute_processor.Run();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Einsum op: An implementation for the input type ",
                         inputs[0]->DataType(), " is not supported yet");
}

}  // namespace cuda

}  // namespace onnxruntime
