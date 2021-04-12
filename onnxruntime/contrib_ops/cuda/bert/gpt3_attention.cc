// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/gpt3_attention.h"
#include "contrib_ops/cuda/bert/gpt3_attention_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Gpt3Attention,                                              \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gpt3Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Gpt3Attention<T>::Gpt3Attention(const OpKernelInfo& info) : CudaKernel(info) {}

template <typename T>
Status Gpt3Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  Tensor* output = context->Output(0, query->Shape());

  size_t element_size = sizeof(T);
  size_t element_count = query->Shape().Size();

  if (!LaunchGpt3AttentionKernel(
        Stream(),
        output->template MutableData<T>(),
        query->template Data<T>(),
        key->template Data<T>(),
        value->template Data<T>(),
        static_cast<int>(element_count),
        element_size)) {
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
