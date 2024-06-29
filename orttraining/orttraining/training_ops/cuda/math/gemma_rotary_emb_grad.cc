// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "orttraining/training_ops/cuda/math/gemma_rotary_emb_grad.h"
#include "orttraining/training_ops/cuda/math/gemma_rotary_emb_grad_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, U)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GemmaRotaryEmbeddingGrad,                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()), \
      GemmaRotaryEmbeddingGrad<T, U>);

REGISTER_KERNEL_TYPED(MLFloat16, float)
REGISTER_KERNEL_TYPED(float, float)

template <typename T, typename U>
GemmaRotaryEmbeddingGrad<T, U>::GemmaRotaryEmbeddingGrad(const OpKernelInfo& info) : CudaKernel(info) {
}

template <typename T, typename U>
Status GemmaRotaryEmbeddingGrad<T, U>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* go0 = context->Input<Tensor>(0);
  const Tensor* go1 = context->Input<Tensor>(1);
  const Tensor* emb = context->Input<Tensor>(2);


  const auto& emb_dims = emb->Shape().GetDims();
  const auto& q_dims = go0->Shape().GetDims();
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

  Tensor* q_grad = context->Output(0, q_dims);
  Tensor* q_rot_grad = context->Output(1, q_dims);
  Tensor* k_grad = context->Output(2, q_dims);
  Tensor* k_rot_grad = context->Output(3, q_dims);

  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;
  return LaunchGemmaRotaryEmbeddingGradKernel<CudaT>(
      Stream(context),
      reinterpret_cast<CudaT*>(q_grad->template MutableData<T>()),
      reinterpret_cast<CudaT*>(q_rot_grad->template MutableData<T>()),
      reinterpret_cast<CudaT*>(k_grad->template MutableData<T>()),
      reinterpret_cast<CudaT*>(k_rot_grad->template MutableData<T>()),
      reinterpret_cast<const CudaT*>(go0->template Data<T>()),
      reinterpret_cast<const CudaT*>(go1->template Data<T>()),
      reinterpret_cast<const CudaU*>(emb->template Data<U>()),
      batch_size,
      num_heads,
      seq_len,
      dim);
}

}  // namespace cuda
}  // namespace onnxruntime
