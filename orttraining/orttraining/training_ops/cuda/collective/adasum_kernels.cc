// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/collective/adasum_kernels.h"

namespace onnxruntime {
namespace cuda {

AdasumAllReduce::AdasumAllReduce(const OpKernelInfo& info) : NcclKernel(info) {
}

Status AdasumAllReduce::ComputeInternal(OpKernelContext* context) const {
  cudaStream_t stream = nullptr;  // Default stream
  ncclComm_t comm = nccl_->Comm(group_type_);

  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    auto onnx_type = input_tensor->DataType();
    const void* input_data = input_tensor->DataRaw();
    size_t input_count = input_tensor->Shape().Size();

    Tensor* output_tensor = context->Output(i, input_tensor->Shape());
    void* output_data = output_tensor->MutableDataRaw();

    ncclDataType_t dtype = GetNcclDataType(onnx_type);
    NCCL_RETURN_IF_ERROR(ncclAllReduce(input_data, output_data, input_count, dtype, ncclSum, comm, stream));
  }

  return Status::OK();
}

static std::vector<std::pair<int, int>> AliasRange(int start, int end) {
  std::vector<std::pair<int, int>> aliases;
  for (int i = start; i < end; i++) {
    aliases.push_back(std::pair<int, int>(i, i));
  }
  return aliases;
}

ONNX_OPERATOR_KERNEL_EX(
    AdasumAllReduce,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(AliasRange(0, 1024))
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
    AdasumAllReduce);

}  // namespace cuda
}  // namespace onnxruntime
