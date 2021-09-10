// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sequence_construction_with_tensor_and_repeat.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

SequenceConstructionWithTensorAndRepeat::SequenceConstructionWithTensorAndRepeat(
    const OpKernelInfo& info) : CudaKernel(info) {}

Status SequenceConstructionWithTensorAndRepeat::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* repeats_tensor = context->Input<Tensor>(1);
  int64_t repeats = repeats_tensor->Data<int64_t>()[0];

  AllocatorPtr alloc;
  ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
              "SequenceInsert GPU: Unable to get an allocator.");

  TensorSeq* Y = context->Output<TensorSeq>(0);
  ORT_ENFORCE(Y != nullptr, "SequenceInsert GPU: Failed to allocate output tensor sequence.");
  Y->SetType(input->DataType());
  Y->Reserve(static_cast<size_t>(repeats));

  for (int64_t i = 0; i < repeats; ++i) {
    std::unique_ptr<Tensor> tensor_to_be_repeated = Tensor::Create(input->DataType(), input->Shape(), alloc);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(tensor_to_be_repeated->MutableDataRaw(),
                                         input->DataRaw(),
                                         input->SizeInBytes(),
                                         cudaMemcpyDeviceToDevice, Stream()));
    Y->Add(std::move(*tensor_to_be_repeated));
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    SequenceConstructionWithTensorAndRepeat,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    SequenceConstructionWithTensorAndRepeat);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
