// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "parse_past_state.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ParsePastState::ParsePastState(
    const OpKernelInfo& info) : CudaKernel(info) {}

Status ParsePastState::ComputeInternal(OpKernelContext* context) const {
  const TensorSeq* input_past_state = context->Input<TensorSeq>(0);
  auto input_past_state_count = input_past_state->Size();
  bool use_provided_past_state = (input_past_state_count != 0);

  auto output_count = static_cast<size_t>(context->OutputCount());

  // The first output will be boolean output
  if (use_provided_past_state && ((output_count - 1) != input_past_state_count)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expected number of element s in provided input tensor sequence: ",
                           output_count - 1, " Got:", input_past_state_count);
  }

  context->Output(0, {})->MutableData<bool>()[0] = use_provided_past_state;

  const auto* past_state_seed = context->Input<Tensor>(1);

  AllocatorPtr alloc;
  ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
              "SequenceInsert GPU: Unable to get an allocator.");

  for (size_t i = 0; i < output_count - 1; ++i) {
    const Tensor* tensor_to_be_copied = use_provided_past_state ? &input_past_state->Get(i) : past_state_seed;
    auto* output = context->Output(static_cast<int>(i + 1), tensor_to_be_copied->Shape());

    void* output_buffer_ptr = output->MutableDataRaw();
    const void* input_buffer_ptr = tensor_to_be_copied->DataRaw();

    if (output_buffer_ptr != input_buffer_ptr) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_buffer_ptr,
                                           input_buffer_ptr,
                                           tensor_to_be_copied->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    ParsePastState,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<bool>()})
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .SequenceTensorToTensorAlias(0, 1),
    ParsePastState);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
