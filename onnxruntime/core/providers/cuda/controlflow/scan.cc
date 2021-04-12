// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/controlflow/scan.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/framework/ml_value.h"
#include "core/framework/ort_value_tensor_slicer.cc"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

template <>
Scan<8>::Scan(const OpKernelInfo& info) : OpKernel(info), scan_cpu_{static_cast<onnxruntime::Scan<8>*>(CreateOpKernel_CPU_Scan_8(info).release())} {
  scan::detail::DeviceHelpers helpers;

  helpers.set_data_to_zero_func = [](void* data, size_t size_in_bytes) -> Status {
    CUDA_RETURN_IF_ERROR(cudaMemset(data, 0, size_in_bytes));
    return Status::OK();
  };

  // copy into base class
  scan_cpu_->SetDeviceHelpers(helpers);
}

template <>
Scan<9>::Scan(const OpKernelInfo& info) : OpKernel(info), scan_cpu_{static_cast<onnxruntime::Scan<9>*>(CreateOpKernel_CPU_Scan_9(info).release())} {
  scan::detail::DeviceHelpers helpers;

  helpers.transpose_func = [this](const std::vector<size_t>& permutations, const Tensor& input, Tensor& output) {
    // TODO: We construct a Transpose kernel on each call as doing so is fairly lightweight.
    // We could potentially keep a single instance and reuse it if that isn't performant enough.
    const OpKernelInfo& info = OpKernel::Info();
    return cuda::Transpose::DoTranspose(cuda::Transpose(info), permutations, input, output);
  };

  // copy into base class
  scan_cpu_->SetDeviceHelpers(helpers);
}

template <>
Status Scan<8>::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = scan_cpu_->Compute(ctx);
  return status;
}

template <>
Status Scan<9>::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = scan_cpu_->Compute(ctx);
  return status;
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scan,
                                  kOnnxDomain,
                                  8, 8,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'sequence_lens' needs to be on CPU
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                  Scan<8>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scan,
                                  kOnnxDomain,
                                  9, 10,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Scan<9>);

// Opset 11 starts to support Neg Axis.
ONNX_OPERATOR_KERNEL_EX(Scan,
                        kOnnxDomain,
                        11,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                            .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                        Scan<9>);

}  // namespace cuda
}  // namespace onnxruntime
