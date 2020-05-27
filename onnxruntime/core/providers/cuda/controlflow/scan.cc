// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/controlflow/scan.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tensor/transpose.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

template <>
Scan<8>::Scan(const OpKernelInfo& info) : onnxruntime::Scan<8>(info) {
  // We need to cast away the const as PerThreadCublasHandle() is currently a non-const method
  // TODO: Clean up the CUDAExecutionProvider interface to avoid this
  cuda_ep_ = const_cast<CUDAExecutionProvider*>(dynamic_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider()));
  scan::detail::DeviceHelpers helpers;

  helpers.set_data_to_zero_func = [](void* data, size_t size_in_bytes) -> Status {
    CUDA_RETURN_IF_ERROR(cudaMemset(data, 0, size_in_bytes));
    return Status::OK();
  };

  // copy into base class
  SetDeviceHelpers(helpers);
}

template <>
Scan<9>::Scan(const OpKernelInfo& info) : onnxruntime::Scan<9>(info) {
  // We need to cast away the const as PerThreadCublasHandle() is currently a non-const method
  // TODO: Clean up the CUDAExecutionProvider interface to avoid this
  cuda_ep_ = const_cast<CUDAExecutionProvider*>(dynamic_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider()));
  scan::detail::DeviceHelpers helpers;

  helpers.transpose_func = [this](const std::vector<size_t>& permutations, const Tensor& input, Tensor& output) {
    return cuda::Transpose::DoTranspose(cuda_ep_->PerThreadCublasHandle(), permutations, input, output);
  };

  // copy into base class
  SetDeviceHelpers(helpers);
}

template <>
Status Scan<8>::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = onnxruntime::Scan<8>::Compute(ctx);
  return status;
}

template <>
Status Scan<9>::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = onnxruntime::Scan<9>::Compute(ctx);
  return status;
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scan,
                                  kOnnxDomain,
                                  8, 8,
                                  kCudaExecutionProvider,
                                  KernelDefBuilder()
                                      .InputMemoryType<OrtMemTypeCPUInput>(0)  // 'sequence_lens' needs to be on CPU
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                  Scan<8>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scan,
                                  kOnnxDomain,
                                  9, 10,
                                  kCudaExecutionProvider,
                                  KernelDefBuilder()
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Scan<9>);

// Opset 11 starts to support Neg Axis.
ONNX_OPERATOR_KERNEL_EX(Scan,
                        kOnnxDomain,
                        11,
                        kCudaExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                            .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                        Scan<9>);

}  // namespace cuda
}  // namespace onnxruntime
