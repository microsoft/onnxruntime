// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sequence_tensor_splitter.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

SequenceTensorSplitter::SequenceTensorSplitter(
    const OpKernelInfo& info) : CudaKernel(info) {}

Status SequenceTensorSplitter::ComputeInternal(OpKernelContext* context) const {
  const TensorSeq* input_sequence = context->Input<TensorSeq>(0);
  auto input_sequence_count = input_sequence->Size();

  auto output_count = static_cast<size_t>(context->OutputCount());

  // The first output will be boolean output
  if (output_count != input_sequence_count) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expected number of element s in provided input tensor sequence: ",
                           output_count, " Got:", input_sequence_count);
  }

  AllocatorPtr alloc;
  ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
              "SequenceInsert GPU: Unable to get an allocator.");

  for (size_t i = 0; i < output_count; ++i) {
    const Tensor* tensor = context->Input<Tensor>(static_cast<int>(i));
    auto* output = context->Output(static_cast<int>(i), tensor->Shape());

    void* output_buffer_ptr = output->MutableDataRaw();
    const void* input_buffer_ptr = tensor->DataRaw();

    if (output_buffer_ptr != input_buffer_ptr) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_buffer_ptr,
                                           input_buffer_ptr,
                                           tensor->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    SequenceTensorSplitter,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .SequenceTensorToTensorAlias(0, 0),
    SequenceTensorSplitter);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
