// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/gemma_rotary_emb.h"
#include "contrib_ops/cuda/bert/gemma_rotary_emb_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, U)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GemmaRotaryEmbedding,                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()), \
      GemmaRotaryEmbedding<T, U>);

REGISTER_KERNEL_TYPED(MLFloat16, float)

template <typename T, typename U>
GemmaRotaryEmbedding<T, U>::GemmaRotaryEmbedding(const OpKernelInfo& info) : CudaKernel(info) {
}

template <typename T, typename U>
Status GemmaRotaryEmbedding<T, U>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* emb = context->Input<Tensor>(0);
  const Tensor* q = context->Input<Tensor>(1);
  const Tensor* q_rot = context->Input<Tensor>(2);
  const Tensor* k = context->Input<Tensor>(3);
  const Tensor* k_rot = context->Input<Tensor>(4);

  const auto& emb_dims = emb->Shape().GetDims();
  const auto& q_dims = q->Shape().GetDims();
  int batch_size = static_cast<int>(q_dims[0]);
  int num_heads = static_cast<int>(q_dims[1]);
  int seq_len = static_cast<int>(q_dims[2]);
  int dim = static_cast<int>(q_dims[3]);

  // q_dims should be [batch_size, num_heads, seq_len, dim]
  // emb_dims should be [batch_size, seq, dim]
  ORT_ENFORCE(emb_dims.size() == 3, "emb_dims should be 3D");
  ORT_ENFORCE(q_dims.size() == 4, "emb_dims should be 4D");
  ORT_ENFORCE(emb_dims[0] == batch_size, "emb_dims[0] should match q_dims[0]");
  ORT_ENFORCE(emb_dims[1] == seq_len, "emb_dims[1] should match q_dims[2]");
  ORT_ENFORCE(emb_dims[2] == dim, "emb_dims[2] should match q_dims[3]");

  Tensor* output1 = context->Output(0, q_dims);
  Tensor* output2 = context->Output(1, q_dims);

  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;
  return LaunchGemmaRotaryEmbeddingKernel<CudaT>(
      Stream(context),
      reinterpret_cast<CudaT*>(output1->template MutableData<T>()),
      reinterpret_cast<CudaT*>(output2->template MutableData<T>()),
      reinterpret_cast<const CudaU*>(emb->template Data<U>()),
      reinterpret_cast<const CudaT*>(q->template Data<T>()),
      reinterpret_cast<const CudaT*>(q_rot->template Data<T>()),
      reinterpret_cast<const CudaT*>(k->template Data<T>()),
      reinterpret_cast<const CudaT*>(k_rot->template Data<T>()),
      batch_size,
      num_heads,
      seq_len,
      dim);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
