// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gpt3_attention.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cuda/cuda_common.h"

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
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* hidden_state = context->Input<Tensor>(1);
  Tensor* output = context->Output(0, input->Shape());

  auto input_data = (input->template Data<T>());
  auto hidden_state_data = (hidden_state->template Data<T>());
  auto output_data = (output->template MutableData<T>());

  size_t size = input->Shape().NumDimensions();

  for (size_t i = 0; i < size; i++) {
    *output_data++ = *input_data++;
  }

  ORT_UNUSED_PARAMETER(hidden_state_data);
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
