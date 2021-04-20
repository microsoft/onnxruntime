// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/controlflow/loop.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_fwd.h"
#include "core/framework/ml_value.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Loop,
                                  kOnnxDomain,
                                  1, 10,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'M' needs to be on CPU
                                      .InputMemoryType(OrtMemTypeCPUInput, 1)  // 'cond' needs to be on CPU
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Loop);

// zero variadic argument support was added in opset 11. using same implementation as for previous version
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Loop,
                                  kOnnxDomain,
                                  11, 12,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'M' needs to be on CPU
                                      .InputMemoryType(OrtMemTypeCPUInput, 1)  // 'cond' needs to be on CPU
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Loop);

// sequence tensors were also supported in addition to existing support for tensors in opset-13,
// but we do not support sequence tensors in the cuda Loop kernel because there are no ops that handle
// sequence tensors on CUDA and supporting it for Loop doesn't add value while that is the case
ONNX_OPERATOR_KERNEL_EX(Loop,
                        kOnnxDomain,
                        13,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'M' needs to be on CPU
                            .InputMemoryType(OrtMemTypeCPUInput, 1)  // 'cond' needs to be on CPU
                            .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                        Loop);

static Status ConcatenateGpuOutput(void* stream, std::vector<OrtValue>& per_iteration_output,
                                   void* output, ptrdiff_t output_size_in_bytes) {
  const auto& first_output = per_iteration_output.front().Get<Tensor>();
  const auto& per_iteration_shape = first_output.Shape();
  size_t bytes_per_iteration = first_output.SizeInBytes();

  void* cur_output = output;
  for (size_t i = 0, num_iterations = per_iteration_output.size(); i < num_iterations; ++i) {
    auto& ort_value = per_iteration_output[i];
    auto& iteration_data = ort_value.Get<Tensor>();

    // sanity check
    if (bytes_per_iteration != iteration_data.SizeInBytes()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Inconsistent shape in loop output for output. ",
                             " Expected:", per_iteration_shape, " Got:", iteration_data.Shape());
    }

    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cur_output, iteration_data.DataRaw(), bytes_per_iteration,
                                         cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));

    cur_output = static_cast<void*>((static_cast<gsl::byte*>(cur_output) + bytes_per_iteration));
  }

  ORT_ENFORCE(static_cast<gsl::byte*>(cur_output) - static_cast<gsl::byte*>(output) == output_size_in_bytes,
              "Concatenation did not fill output buffer as expected.");

  return Status::OK();
}

Loop::Loop(const OpKernelInfo& info) : onnxruntime::Loop(info) {
  SetConcatOutputFunc(ConcatenateGpuOutput);
  SetComputeStream(static_cast<void*>(info.GetExecutionProvider()->GetComputeStream()));
}

Status Loop::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = onnxruntime::Loop::Compute(ctx);
  return status;
}

}  // namespace cuda
}  // namespace onnxruntime
