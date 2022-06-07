// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/controlflow/loop.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_fwd.h"
#include "core/providers/cuda/cuda_execution_provider.h"

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

// opset-13 supports sequence type for loop carried dependencies
ONNX_OPERATOR_KERNEL_EX(Loop,
                        kOnnxDomain,
                        13,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'M' needs to be on CPU
                            .InputMemoryType(OrtMemTypeCPUInput, 1)  // 'cond' needs to be on CPU
                            .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
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
  // We use the IDataTransfer abstraction to perform copies in the Loop implementation.
  // By default, the GPUDataTransfer class is setup to use the same stream as the EP's compute stream
  // while performing copies to/from CUDA (do_copy_on_default_stream = true). This is good as we wouldn't
  // have to do any explicit syncs between the copy and compute streams.
  // However, there is a user-facing flag that allows users to use a dedicated stream just for copying.
  // To support using Loop for that case, we would have to do a sync between the copy stream and
  // the compute stream to avoid data races. At the very least, we need to expose an interface in IDataTransfer
  // to use a caller provided stream for Loop to provide for the GPUDataTransfer instance to use.
  // Currently, using a dedicated copy stream has larger negative implications (see comment in GPUDataTransfer's
  // constructor implementation), and so it is not in a usable state. When it becomes usable again,
  // we will re-visit this limitation in Loop.
  bool do_copy_on_default_stream = static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider())->DoCopyOnDefaultStream();
  ORT_ENFORCE(do_copy_on_default_stream,
              "Using Loop operator on CUDA while using a dedicated stream for copying "
              "(a stream that is different than the compute stream) is currently not supported");
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
