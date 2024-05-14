// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/restore_padding.h"
#include "contrib_ops/cuda/bert/bert_padding.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      RestorePadding,                                             \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      RestorePadding<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
RestorePadding<T>::RestorePadding(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status RestorePadding<T>::ComputeInternal(OpKernelContext* context) const {
  // shape of inputs:
  //   input:                (total_tokens, hidden_size)
  //   token_offset:         (batch_size, sequence_length)
  // shape of outputs:
  //   output:               (batch_size, sequence_length, hidden_size)

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* token_offset = context->Input<Tensor>(1);

  const auto& dims = input->Shape().GetDims();
  if (dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 2 dimensions, got ",
                           dims.size());
  }
  int64_t total_tokens = dims[0];
  int64_t hidden_size = dims[1];

  const auto& token_offset_dims = token_offset->Shape().GetDims();
  if (token_offset_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'token_offset' is expected to have 2 dimensions, got ",
                           token_offset_dims.size());
  }
  int64_t batch_size = token_offset_dims[0];
  int64_t sequence_length = token_offset_dims[1];

  TensorShapeVector output_shape(3);
  output_shape[0] = batch_size;
  output_shape[1] = sequence_length;
  output_shape[2] = hidden_size;
  Tensor* output = context->Output(0, output_shape);

  typedef typename ToCudaType<T>::MappedType CudaT;
  LaunchRestorePadding<CudaT>(
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<const CudaT*>(input->Data<T>()),
      token_offset->Data<int>(),
      static_cast<int>(total_tokens),
      static_cast<int>(hidden_size),
      static_cast<int>(batch_size),
      static_cast<int>(sequence_length),
      Stream(context));

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
