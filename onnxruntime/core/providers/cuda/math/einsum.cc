// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum.h"
#include "core/providers/cpu/math/einsum_utils/einsum_compute_preprocessor.h"
#include "core/providers/cpu/math/einsum_utils/einsum_typed_compute_processor.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/math/matmul.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Einsum,
    kOnnxDomain,
    12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<float>(),
                                 DataTypeImpl::GetTensorType<double>(),
                                 DataTypeImpl::GetTensorType<MLFloat16>()}),
    Einsum);

Status Einsum::ComputeInternal(OpKernelContext* context) const {
  int num_inputs = context->InputCount();
  if (num_inputs == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Einsum op: at least one input is required.");
  }

  std::vector<const Tensor*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(context->Input<Tensor>(i));
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto tp = static_cast<concurrency::ThreadPool*>(nullptr);

  EinsumEquationPreprocessor einsum_equation_preprocessor(*einsum_equation_preprocessor_);

  auto ort_stream = GetOrtStream(context);
  EinsumOp::EinsumCudaAssets einsum_cuda_assets(
      ort_stream,
      GetAllocationStream(context),
      GetDeviceProp(),
      GetCublasHandle(context),
      GetCudnnHandle(context),
      allocator,
      UseTF32());

  // `CreateTensor` in the shared device helper interface cannot take the assets (its signature is
  // part of the CPU provider bridge), so bind them here - the CUDA implementation needs them to
  // allocate the intermediate on the stream that will use it.
  EinsumOp::DeviceHelpers::CreateTensor create_tensor_func =
      [&einsum_cuda_assets](const DataTypeImpl* type, const TensorShape& shape, AllocatorPtr allocator_ptr) {
        return EinsumOp::DeviceHelpers::CudaDeviceHelpers::CreateTensor(type, shape, std::move(allocator_ptr),
                                                                        &einsum_cuda_assets);
      };

  EinsumComputePreprocessor einsum_compute_preprocessor(einsum_equation_preprocessor, inputs, allocator,
                                                        &einsum_cuda_assets);
  einsum_compute_preprocessor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Diagonal,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                               create_tensor_func);

  ORT_RETURN_IF_ERROR(einsum_compute_preprocessor.Run());

  const auto& first_input_tensor = inputs[0];

  if (first_input_tensor->IsDataType<float>()) {
    EinsumTypedComputeProcessor<float> einsum_compute_processor(context, allocator, tp, nullptr, einsum_compute_preprocessor, &einsum_cuda_assets);
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<float>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<float>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ZeroBuffer,
                                              create_tensor_func);
    return einsum_compute_processor.Run();

  } else if (first_input_tensor->IsDataType<double>()) {
    EinsumTypedComputeProcessor<double> einsum_compute_processor(context, allocator, tp, nullptr, einsum_compute_preprocessor, &einsum_cuda_assets);
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<double>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<double>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ZeroBuffer,
                                              create_tensor_func);
    return einsum_compute_processor.Run();

  } else if (first_input_tensor->IsDataType<MLFloat16>()) {
    EinsumTypedComputeProcessor<MLFloat16> einsum_compute_processor(context, allocator, tp, nullptr, einsum_compute_preprocessor, &einsum_cuda_assets);
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<MLFloat16>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<MLFloat16>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ZeroBuffer,
                                              create_tensor_func);
    return einsum_compute_processor.Run();

  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "Einsum op: Unsupported/unimplemented data type encountered: ",
                           first_input_tensor->DataType());
  }
}

}  // namespace cuda
}  // namespace onnxruntime
