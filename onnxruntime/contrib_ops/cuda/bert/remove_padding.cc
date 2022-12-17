// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/remove_padding.h"
#include "contrib_ops/cuda/bert/bert_padding.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      RemovePadding,                                                           \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .OutputMemoryType(OrtMemTypeCPUOutput, 3) /*max_token_count on CPU*/ \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      RemovePadding<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
RemovePadding<T>::RemovePadding(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status RemovePadding<T>::ComputeInternal(OpKernelContext* context) const {
  // shape of inputs:
  //   input:                   (batch_size, sequence_length, hidden_size)
  //   sequence_token_count:    (batch_size)
  // shape of outputs:
  //   output:                  (total_tokens, hidden_size)
  //   token_offset:            (batch_size, sequence_length)
  //   cumulated_seq_len:       (batch_size + 1)
  //   max_token_count:         (1)

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* sequence_token_count = context->Input<Tensor>(1);

  const auto& dims = input->Shape().GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 3 dimensions, got ",
                           dims.size());
  }
  int64_t batch_size = dims[0];
  int64_t sequence_length = dims[1];
  int64_t hidden_size = dims[2];

  auto token_count_buffer = GetScratchBuffer<int>(2, context->GetComputeStream());

  TensorShapeVector token_offset_shape(2);
  token_offset_shape[0] = batch_size;
  token_offset_shape[1] = sequence_length;
  Tensor* token_offset = context->Output(1, token_offset_shape);

  TensorShapeVector cumulated_seq_len_shape(1);
  cumulated_seq_len_shape[0] = batch_size + static_cast<int64_t>(1);
  Tensor* cumulated_seq_len = context->Output(2, cumulated_seq_len_shape);

  LaunchGetTokenOffset(token_count_buffer.get(),
                       token_offset->MutableData<int>(),
                       cumulated_seq_len->MutableData<int>(),
                       sequence_token_count->Data<int>(),
                       static_cast<int>(batch_size),
                       static_cast<int>(sequence_length),
                       Stream(context));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  // Copy token_count to CPU
  auto pinned_buffer = AllocateBufferOnCPUPinned<int>(2);
  int* token_count_pinned = pinned_buffer.get();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(token_count_pinned,
                                       token_count_buffer.get(),
                                       sizeof(int) * 2,
                                       cudaMemcpyDeviceToHost,
                                       Stream(context)));
  // Wait until token_count is copied to host.
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(Stream(context)));
  int total_token_count = token_count_pinned[0];
  int max_token_count = token_count_pinned[1];

  TensorShapeVector output_shape(2);
  output_shape[0] = static_cast<int64_t>(total_token_count);
  output_shape[1] = hidden_size;
  Tensor* output = context->Output(0, output_shape);

  TensorShapeVector max_token_count_shape(1);
  max_token_count_shape[0] = 1;
  Tensor* max_token_count_tensor = context->Output(3, max_token_count_shape);
  max_token_count_tensor->MutableData<int>()[0] = max_token_count;

  typedef typename ToCudaType<T>::MappedType CudaT;
  LaunchRemovePadding<CudaT>(
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<const CudaT*>(input->Data<T>()),
      token_offset->Data<int>(),
      total_token_count,
      static_cast<int>(hidden_size),
      Stream(context));

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
