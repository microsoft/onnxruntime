// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/rotary_embedding.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      RotaryEmbedding,                                                  \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()), \
      RotaryEmbedding<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
Status RotaryEmbedding<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* position_ids = context->Input<Tensor>(1);
  const Tensor* cos_cached = context->Input<Tensor>(2);
  const Tensor* sin_cached = context->Input<Tensor>(3);
  const Tensor* past_key = context->Input<Tensor>(4);

  // input shape is [batch, num_head, seqlen, head_dim]
  const TensorShape& in_shape = input->Shape();
  int64_t batch_size = in_shape[0];
  int64_t num_heads = in_shape[1];
  int64_t seqlen = in_shape[2];
  int64_t head_dim = in_shape[3];
  int64_t seqlen_with_past = seqlen;

  if (past_key != nullptr) {
    // past_key with shape [batch, num_head, past_seqlen, head_dim]
    const TensorShape& past_shape = past_key->Shape();
    seqlen_with_past += past_shape[2];
  }

  Tensor* output = context->Output(0, input->Shape());

  typedef typename ToCudaType<T>::MappedType CudaT;

  const CudaT* input_buffer = reinterpret_cast<const CudaT*>(input->Data<T>());
  const int64_t* pos_buffer = reinterpret_cast<const int64_t*>(position_ids->Data<int64_t>());

  // cos and sin wit shape [1, 1, max_seq_len, head_dim]
  // only use [1, 1, :seqlen, head_dim] of them, that is [0 ~ seqlen*head_dim)
  const CudaT* cos_buffer = reinterpret_cast<const CudaT*>(cos_cached->Data<T>());
  const CudaT* sin_buffer = reinterpret_cast<const CudaT*>(sin_cached->Data<T>());

  return LaunchRotaryEmbeddingKernel<CudaT>(
      context->GetComputeStream(),
      input_buffer, batch_size, num_heads, seqlen, head_dim, seqlen_with_past,
      pos_buffer, cos_buffer, sin_buffer,
      reinterpret_cast<CudaT*>(output->MutableData<T>()));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
