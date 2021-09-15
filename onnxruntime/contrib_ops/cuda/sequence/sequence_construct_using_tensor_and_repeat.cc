// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sequence_construct_using_tensor_and_repeat.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

SequenceConstructUsingTensorAndRepeat::SequenceConstructUsingTensorAndRepeat(
    const OpKernelInfo& info) : CudaKernel(info) {}

Status SequenceConstructUsingTensorAndRepeat::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  auto repeat = context->Input<Tensor>(1)->Data<int64_t>()[0];
  if (repeat <= 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Repeat needs to be positive");
  }

  MLDataType dtype = input->DataType();

  TensorSeq* output = context->Output<TensorSeq>(0);
  output->SetType(dtype);
  output->Reserve(static_cast<size_t>(repeat));

  if (repeat == 0) {
    return Status::OK();
  }

  AllocatorPtr alloc;
  ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
              "SequenceConstructUsingTensorAndRepeat GPU: Unable to get an allocator.");

  for (int64_t i = 0; i < repeat; ++i) {
    std::unique_ptr<Tensor> copy = Tensor::Create(input->DataType(),
                                                  input->Shape(), alloc);

    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(copy->MutableDataRaw(),
                                         input->DataRaw(),
                                         input->SizeInBytes(),
                                         cudaMemcpyDeviceToDevice, Stream()));

    output->Add(std::move(*copy));  // Add will check for type consistency
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    SequenceConstructUsingTensorAndRepeat,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    SequenceConstructUsingTensorAndRepeat);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
