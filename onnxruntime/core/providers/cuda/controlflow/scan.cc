// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/controlflow/scan.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tensor/transpose.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

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
                                      .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                  Scan<9>);

// Opset 11 starts to support Neg Axis.
ONNX_OPERATOR_KERNEL_EX(Scan,
                        kOnnxDomain,
                        11,
                        kCudaExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                            .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                        Scan<9>);

template <>
Scan<8>::Scan(const OpKernelInfo& info) : onnxruntime::Scan<8>(info) {
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
  scan::detail::DeviceHelpers helpers;

  helpers.transpose_func = [this](const std::vector<size_t>& permutations, const Tensor& input, Tensor& output) {
    // TODO: We construct a Transpose kernel on each call as doing so is fairly lightweight.
    // We could potentially keep a single instance and reuse it if that isn't performant enough.
    const OpKernelInfo& info = OpKernel::Info();
    return cuda::Transpose::DoTranspose(cuda::Transpose(info), permutations, input, output);
  };

  // copy into base class
  SetDeviceHelpers(helpers);
}

Status Scan<8>::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = onnxruntime::Scan<8>::Compute(ctx);
  return status;
}

Status Scan<9>::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = onnxruntime::Scan<9>::Compute(ctx);
  return status;
}

}  // namespace cuda
}  // namespace onnxruntime
